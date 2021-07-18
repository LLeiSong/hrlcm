# Title     : Script to make guess labels
# Objective : To use random forest guesser
#             to get new labels.
# Created by: Lei Song
# Created on: 02/24/21

#######################
##  Step 1: Setting  ##
#######################
message('Step1: Setting')

library(sf)
library(here)
library(terra)
library(dplyr)
library(ramify)
library(ranger)
library(parsnip)
library(tidymodels)
library(stringr)
library(glue)
library(parallel)
library(rgrass7)

#############################
###  Step 2: Preparation  ###
#############################
message('Step 2: Preparation')

# Load model
message("--Load model")
load(here('data/north/guess_rf_md.rda'))

# Define function to do prediction
make_pred <- function(tile_id,
                      sample_tiles,
                      tiles,
                      size_sub_tile = 512,
                      skip_class = 8,
                      img_dir,
                      dst_dir = 'results/north/guess_labels'){
    message(paste0('--', tile_id))
    
    # Get image
    imgs <- rast(file.path(img_dir, paste0(tile_id, '.tif')))
    
    # Subset vectors for whole tile
    tile <- tiles %>% 
        filter(tile == tile_id) %>% 
        st_buffer(0.01)
    sub_tiles <- sample_tiles %>% 
        filter(tile == tile_id)
    crs_mer <- crs(imgs)
    road <- roads %>% 
        st_intersection(tile) %>% 
        st_cast('MULTILINESTRING') %>% 
        st_transform(crs = crs_mer)
    waterbody <- waterbodies %>% 
        st_intersection(tile) %>% 
        st_cast('MULTIPOLYGON') %>% 
        st_transform(crs = crs_mer)
    building <- buildings %>% 
        st_intersection(tile) %>% 
        st_cast('MULTIPOLYGON') %>% 
        st_transform(crs = crs_mer)
    
    # Loop on sampled sub-tiles to do prediction
    tiles_layout <- aggregate(imgs$os_band1, fact = size_sub_tile)
    values(tiles_layout) <- 1:(4096 / size_sub_tile)^2
    # A bit hard coded for default
    skip_class <- paste0('.pred_', skip_class)
    
    # Make prediction and refine the guess labels
    lapply(sub_tiles$index, function(index){
        # Make mask
        mask <- copy(tiles_layout)
        mask <- mask == index
        mask[mask == 0] <- NA
        mask <- disaggregate(mask, fact = size_sub_tile)
        
        # Subset images
        imgs_sub <- imgs * mask
        imgs_sub <- terra::trim(imgs_sub)
        rm(mask)
        
        # Do prediction
        scores <- predict(imgs_sub, guess_rf_md, type = "prob", na.rm = T)
        writeRaster(
            scores, 
            here(glue('{dst_dir}/scores_{tile_id}_{index}.tif')))
        scores[[skip_class]] <- 0
        pred <- values(scores); pred[is.na(pred)] <- 0
        pred <- argmax(pred)
        classes <- scores[[1]]
        values(classes) <- pred
        rm(scores, pred); gc()
        
        # Add vectors
        # Set up GRASS GIS
        message('------Add OSM layers')
        crs_mer <- crs(imgs, proj4 = T)
        gisBase <- '/Applications/GRASS-7.9.app/Contents/Resources'
        initGRASS(gisBase = gisBase,
                  home = tempdir(),
                  gisDbase = tempdir(),  
                  mapset = 'PERMANENT', 
                  location = 'segments', 
                  override = TRUE)
        execGRASS("g.proj", flags = "c", 
                  proj4 = crs_mer)
        execGRASS('r.in.gdal', flags = c("o", "overwrite"),
                  input = file.path(
                      dst_dir, 
                      glue('scores_{tile_id}_{index}.tif')),
                  band = 1,
                  output = "score")
        execGRASS("g.region", raster = "score")
        
        use_sf()
        if (nrow(road) > 0){
            writeVECT(road, 'roads', v.in.ogr_flags = 'overwrite')
            execGRASS('v.to.rast', flags = c("overwrite"),
                      parameters = list(input = 'roads', 
                                        output = 'roads',
                                        use = 'val',
                                        value = 7))
            roads_path <- tempfile()
            execGRASS('r.out.gdal', flags = c("m", "overwrite"),
                      output = roads_path,
                      input = "roads")
            fill_roads <- rast(roads_path)
        } else {
            fill_roads <- NULL
        }; rm(road)
        
        if (nrow(waterbody) > 0){
            writeVECT(waterbody, 'waterbodies', v.in.ogr_flags = 'overwrite')
            execGRASS('v.to.rast', flags = c("overwrite"),
                      parameters = list(input = 'waterbodies', 
                                        output = 'waterbodies',
                                        use = 'val',
                                        value = 5))
            waterbody_path <- tempfile()
            execGRASS('r.out.gdal', flags = c("m", "overwrite"),
                      output = waterbody_path,
                      input = "waterbodies")
            fill_waterbodies <- rast(waterbody_path)
        } else {
            fill_waterbodies <- NULL
        }; rm(waterbody)
        
        if (nrow(building) > 0){
            writeVECT(building, 'buildings', v.in.ogr_flags = 'overwrite')
            execGRASS('v.to.rast', flags = c("overwrite"),
                      parameters = list(input = 'buildings', 
                                        output = 'buildings',
                                        use = 'val',
                                        value = 6))
            building_path <- tempfile()
            execGRASS('r.out.gdal', flags = c("m", "overwrite"),
                      output = building_path,
                      input = "buildings")
            fill_buildings <- rast(building_path)
        } else {
            fill_buildings <- NULL
        }; rm(building)
        
        # Gather results
        fills <- compact(list(roads = fill_roads,
                              buildings = fill_buildings,
                              waterbodies = fill_waterbodies))
        rm(fill_roads, fill_buildings, fill_waterbodies)
        
        # Replace values
        if (length(fills) > 0){
            for (i in 1:length(fills)){
                mask_vct <- fills[[i]]
                mask_vct <- is.na(mask_vct)
                classes <- classes * mask_vct
                classes <- cover(classes, fills[[i]], values = 0)
                rm(mask_vct)
            }
        }
        
        # Save out
        names(classes) <- 'class'
        writeRaster(classes, 
                    file.path(dst_dir, 
                              glue('guess_{tile_id}_{index}.tif')),
                    wopt = list(datatype = 'INT1U',
                                gdal=c("COMPRESS=LZW")))
    })
}

#############################
#  Step 3: Make prediction  #
#############################
message('Step 3: Make prediction')

# Directories
stack_dir <- '/Volumes/elephant/pred_stack'
labels_dir <- here('results/north/guess_labels')
if (!dir.exists(labels_dir)) dir.create(labels_dir)

# Read vectors
tiles <- read_sf('data/geoms/tiles_nicfi_north.geojson')
roads <- read_sf(here('data/osm/roads.geojson'))
roads <- roads %>%
    filter(fclass %in% c('primary', 'secondary', 'tertiary',
                         'trunk', 'primary_link', 'secondary_link',
                         'tertiary_link', 'rail'))
waterbodies <- read_sf(here('data/osm/waterbodies.geojson'))
buildings <- read_sf(here('data/osm/buildings.geojson'))

# Cut the tiles and make 8 samples
# 6 for train, 2 for validate
n_sample <- 6
indices <- matrix(1:64, 8, 8)
indices <- indices[, ncol(indices):1]
sample_tiles <- do.call(rbind, lapply(1:nrow(tiles), function(n){
    tile <- tiles %>% slice(n)
    set.seed(n)
    tiles_grids <- st_make_grid(
        tile, 
        n = c(8, 8)) %>% 
        st_sf() %>% 
        mutate(tile = tile$tile,
               index = as.vector(indices),
               score = 0,  hardiness = 1, # 1 - 5
               pass = 'yes', modify = 'no',
               comment = '') %>% 
        slice(sample(1:nrow(.), n_sample))
}))
st_write(sample_tiles, here('results/north/catalog_sample_tiles.geojson'))

mclapply(unique(sample_tiles$tile), function(tile_id){
    make_pred(tile_id = tile_id,
              sample_tiles = sample_tiles,
              tiles = tiles,
              img_dir = stack_dir)
}, mc.cores = 4)