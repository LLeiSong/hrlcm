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
                      road, # sf object
                      waterbody, # sf object
                      building, # sf object
                      size_sub_tile = 512,
                      skip_class = 8,
                      img_dir,
                      dst_dir = 'results/north/guess_labels'){
    message(paste0('--', tile_id))
    
    # Get image
    imgs <- rast(file.path(img_dir, paste0(tile_id, '.tif')))
    
    # Subset vectors for whole tile
    sub_tiles <- sample_tiles %>% 
        filter(tile == tile_id)
    
    # Loop on sampled sub-tiles to do prediction
    tiles_layout <- aggregate(imgs$os_band1, fact = size_sub_tile)
    values(tiles_layout) <- 1:(4096 / size_sub_tile)^2
    # A bit hard coded for default
    skip_class <- paste0('.pred_', skip_class)
    
    # Cut image
    imgs <- lapply(sub_tiles$index, function(index){
        # Make mask
        mask <- copy(tiles_layout)
        mask <- mask == index
        mask[mask == 0] <- NA
        mask <- disaggregate(mask, fact = size_sub_tile)
        
        # Subset images
        imgs_sub <- imgs * mask
        imgs_sub <- terra::trim(imgs_sub)
        rm(mask); imgs_sub
    })
    names(imgs) <- sub_tiles$index
    
    # Transform vectors
    crs_mer <- crs(imgs[[1]])
    road <- road %>% 
        st_cast('MULTILINESTRING') %>% 
        st_transform(crs = crs_mer)
    waterbody <- waterbody %>% 
        st_cast('MULTIPOLYGON') %>% 
        st_transform(crs = crs_mer)
    building <- building %>% 
        st_cast('MULTIPOLYGON') %>% 
        st_transform(crs = crs_mer)
    
    # Make prediction and refine the guess labels
    lapply(sub_tiles$index, function(index){
        message(paste0('----', index))
        
        # Do prediction
        message('------Make scores and labels')
        imgs_sub <- imgs[[as.character(index)]]
        scores <- predict(imgs_sub, guess_rf_md, type = "prob", na.rm = T)
        writeRaster(
            scores, 
            here(glue('{dst_dir}/scores_{tile_id}_{index}.tif')),
            overwrite = T)
        scores[[skip_class]] <- 0
        pred <- values(scores); pred[is.na(pred)] <- 0
        pred <- argmax(pred)
        classes <- scores[[1]]
        values(classes) <- pred
        rm(scores, pred); gc()
        
        # Add vectors
        # Set up GRASS GIS
        message('------Add OSM layers')
        crs_mer <- crs(imgs_sub, proj4 = T)
        gisBase <- '/Applications/GRASS-7.9.app/Contents/Resources'
        initGRASS(gisBase = gisBase,
                  home = tempdir(),
                  gisDbase = tempdir(),  
                  mapset = 'PERMANENT', 
                  location = glue('segments_{tile_id}_{index}'), 
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
                                        value = 3))
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
        message('------Save out')
        names(classes) <- 'class'
        writeRaster(classes, 
                    file.path(dst_dir, 
                              glue('guess_{tile_id}_{index}.tif')),
                    overwrite = T,
                    wopt = list(datatype = 'INT1U',
                                gdal=c("COMPRESS=LZW")))
    })
}

#############################
#  Step 3: Make prediction  #
#############################
message('Step 3: Make prediction')

# Directories
message("--Set directories")
stack_dir <- '/Volumes/elephant/pred_stack'
labels_dir <- here('results/north/guess_labels')
if (!dir.exists(labels_dir)) dir.create(labels_dir)

# Read vectors
message("--Read vectors")
tiles <- read_sf('data/geoms/tiles_nicfi_north.geojson')
roads <- read_sf(here('data/osm/roads.geojson'))
roads <- roads %>%
    filter(fclass %in% c('primary', 'secondary', 'tertiary',
                         'trunk', 'primary_link', 'secondary_link',
                         'tertiary_link', 'rail'))
roads <- st_join(roads, tiles)
waterbodies <- read_sf(here('data/osm/waterbodies.geojson'))
waterbodies <- st_join(waterbodies, tiles)
buildings <- read_sf(here('data/osm/buildings.geojson'))
buildings <- st_join(buildings, tiles)

# Cut the tiles and make 4 samples
# 3 for train, 1 for validate
message('--Generate catalog')
n_sample <- 4
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
# st_write(sample_tiles, here('results/north/catalog_sample_tiles.geojson'))

# Filter finished ones [IN CASE]
# tiles_finished <- list.files(labels_dir, pattern = 'guess')
# tiles_finished <- str_extract(tiles_finished,
#                               '[0-9]+-[0-9]+_[0-9]+')
# sample_tiles <- sample_tiles %>% 
#     mutate(fname = paste(tile, index, sep = '_')) %>% 
#     filter(!fname %in% tiles_finished) %>% 
#     dplyr::select(-fname)

message('--Start generating')
lapply(unique(sample_tiles$tile), function(tile_id){
    road <- roads %>% 
        filter(tile == tile_id)
    waterbody <- waterbodies %>% 
        filter(tile == tile_id)
    building <- buildings %>% 
        filter(tile == tile_id)
    make_pred(tile_id = tile_id,
              sample_tiles = sample_tiles,
              road = road,
              waterbody = waterbody,
              building = building,
              img_dir = stack_dir)
})