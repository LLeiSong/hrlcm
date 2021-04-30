# Title     : Script to get training data
# Objective : To extract satellite values  
#             from a bunch of images.
# Created by: Lei Song
# Created on: 02/24/21

#######################
##  Step 1: Setting  ##
#######################
message('Step1: Setting')

## Load packages
library(here)
library(terra)
library(parallel)
library(sf)
library(dplyr)
library(tidyr)
library(stringr)
library(rgrass7)

####################################
##  Step 2: Load related dataset  ##
####################################
message('Step 2: Load related dataset')

## Get samples
samples <- st_read(here('data/north/samples_all.geojson'))
tiles_north <- st_read(here('data/geoms/tiles_nicfi_north.geojson'))
samples <- st_join(samples, tiles_north)
save(samples, file = here('data/north/samples.rda'))

## Read file names
plt_path <- '/Volumes/elephant/plt_nicfi'
plt_nms <- list.files(plt_path, full.names = T)
s1_path <- '/Volumes/elephant/sentinel1_hr_coefs'
s1_nms <- list.files(s1_path, full.names = T)

####################################
##  Step 3: Prepare image stacks  ##
####################################
message('Step 3: Prepare image stacks')

## Define the function to make image stacks
get_img_wdist <- function(tile_nm, tiles_north,
                    rivers_all, roads_all, 
                    buildings_all){
    message(tile_nm)
    
    # read Planet images
    message('--Planet basemap')
    plt_nms_tile <- grep(tile_nm, plt_nms, value = TRUE)
    plt_os <- rast(grep('2017-12_2018-05', 
                        plt_nms_tile, value = TRUE)) %>% 
        subset(1:4)
    names(plt_os) <- paste0('band', 1:4)
    plt_os$ndvi <- (plt_os$band4 - plt_os$band3) / (plt_os$band4 + plt_os$band3)
    plt_os$evi <- 2.5 * ((plt_os$band4 - plt_os$band3) / 
                             (plt_os$band4 + 2.4 * plt_os$band3 + 1))
    plt_os$savi <- ((plt_os$band4 - plt_os$band3) / 
                        (plt_os$band4 + plt_os$band3 + 1)) * 2
    plt_os$arvi <- (plt_os$band4 - (2 * plt_os$band3) + plt_os$band1) / 
        (plt_os$band4 + (2 * plt_os$band3) + plt_os$band1)
    plt_gs <- rast(grep('2018-06_2018-11', 
                        plt_nms_tile, value = TRUE)) %>% 
        subset(1:4)
    names(plt_gs) <- paste0('band', 1:4)
    plt_gs$ndvi <- (plt_gs$band4 - plt_gs$band3) / (plt_gs$band4 + plt_gs$band3)
    plt_gs$evi <- 2.5 * ((plt_gs$band4 - plt_gs$band3) / 
                             (plt_gs$band4 + 2.4 * plt_gs$band3 + 1))
    plt_gs$savi <- ((plt_gs$band4 - plt_gs$band3) / 
                        (plt_gs$band4 + plt_gs$band3 + 1)) * 2
    plt_gs$arvi <- (plt_gs$band4 - (2 * plt_gs$band3) + plt_gs$band1) / 
        (plt_gs$band4 + (2 * plt_gs$band3) + plt_gs$band1)
    plts <- c(plt_os, plt_gs)
    names(plts) <- c(paste('os', names(plt_os), sep = '_'),
                     paste('gs', names(plt_gs), sep = '_'))
    rm(plt_os, plt_gs); gc()
    
    # Read S1 images
    message('--Sentinel-1 images')
    s1_nms_tile <- grep(tile_nm, s1_nms, value = TRUE)
    s1_vv <- rast(grep('VV', s1_nms_tile, value = TRUE))
    s1_vv <- resample(s1_vv, plts$os_band1)
    names(s1_vv) <- paste0('vv', 1:6)
    s1_vh <- rast(grep('VH', s1_nms_tile, value = TRUE))
    s1_vh <- resample(s1_vh, plts$os_band1)
    names(s1_vh) <- paste0('vh', 1:6)
    
    # Get distance layers
    message('--Distance layers')
    temp_path <- glue::glue('data/north/temp_{tile_nm}')
    dir.create(temp_path)
    
    tile_this <- tiles_north %>% 
        filter(tile == tile_nm) %>% 
        st_buffer(0.2)
    
    # rivers <- rivers_all %>% 
    #     st_intersection(tile_this) %>% 
    #     st_collection_extract('LINESTRING') %>% 
    #     st_transform(crs(plts, proj4 = TRUE))
    
    roads <- roads_all %>% 
        st_intersection(tile_this) %>% 
        st_collection_extract('LINESTRING') %>% 
        st_transform(crs(plts, proj4 = TRUE))
    
    buildings <- buildings_all %>% 
        st_intersection(tile_this) %>% 
        st_collection_extract('POLYGON') %>% 
        st_transform(crs(plts, proj4 = TRUE))
    
    if (nrow(roads) == 0 | nrow(buildings) == 0){
        tile_this <- tiles_north %>% 
            filter(tile == tile_nm) %>% 
            st_buffer(1)
        # rivers <- rivers_all %>% 
        #     st_intersection(tile_this) %>% 
        #     st_collection_extract('LINESTRING') %>% 
        #     st_transform(crs(plts, proj4 = TRUE))
        
        roads <- roads_all %>% 
            st_intersection(tile_this) %>% 
            st_collection_extract('LINESTRING') %>% 
            st_transform(crs(plts, proj4 = TRUE))
        
        buildings <- buildings_all %>% 
            st_intersection(tile_this) %>% 
            st_collection_extract('POLYGON') %>% 
            st_transform(crs(plts, proj4 = TRUE))
        
        terra::expand(plts$os_band1, 10000, 
                      filename = file.path(temp_path,
                                           glue::glue('temp_{tile_nm}.tif')),
                      wopt = list(gdal=c("COMPRESS=LZW")))
    } else{
        terra::expand(plts$os_band1, 2000, 
                      filename = file.path(temp_path,
                                           glue::glue('temp_{tile_nm}.tif')),
                      wopt = list(gdal=c("COMPRESS=LZW")))
    }
    
    gisBase <- '/Applications/GRASS-7.9.app/Contents/Resources'
    initGRASS(gisBase = gisBase,
              home = temp_path,
              gisDbase = temp_path,  
              mapset = 'PERMANENT', 
              location = 'dist', 
              override = TRUE)
    execGRASS("g.proj", flags = "c", 
              proj4 = crs(plts, proj4 = TRUE))
    execGRASS('r.in.gdal', flags = c("o", "overwrite"),
              input = file.path(temp_path,
                                glue::glue('temp_{tile_nm}.tif')),
              band = 1,
              output = "temp")
    execGRASS("g.region", raster = "temp")
    
    # # rivers
    # writeVECT(rivers, 'rivers', v.in.ogr_flags = 'overwrite')
    # execGRASS('v.to.rast', flags = c("overwrite"),
    #           parameters = list(input = 'rivers', 
    #                             output = 'rivers',
    #                             use = 'val'))
    # execGRASS('r.grow.distance', flags = c("overwrite"),
    #           parameters = list(input = 'rivers', 
    #                             distance = 'rivers_dist'))
    # execGRASS('r.out.gdal', flags = c("c", "overwrite"),
    #           parameters = list(input = 'rivers_dist', 
    #                             output = file.path(temp_path, 
    #                                                'dist_rivers.tif')))
    # roads
    writeVECT(roads, 'roads', v.in.ogr_flags = 'overwrite')
    execGRASS('v.to.rast', flags = c("overwrite"),
              parameters = list(input = 'roads', 
                                output = 'roads',
                                use = 'val'))
    execGRASS('r.grow.distance', flags = c("overwrite"),
              parameters = list(input = 'roads', 
                                distance = 'roads_dist'))
    execGRASS('r.out.gdal', flags = c("c", "overwrite"),
              parameters = list(input = 'roads_dist', 
                                output = file.path(temp_path, 
                                                   'dist_roads.tif')))
    
    # buildings
    writeVECT(buildings, 'buildings', v.in.ogr_flags = 'overwrite')
    execGRASS('v.to.rast', flags = c("overwrite"),
              parameters = list(input = 'buildings', 
                                output = 'buildings',
                                use = 'val'))
    execGRASS('r.grow.distance', flags = c("overwrite"),
              parameters = list(input = 'buildings', 
                                distance = 'buildings_dist'))
    execGRASS('r.out.gdal', flags = c("c", "overwrite"),
              parameters = list(input = 'buildings_dist', 
                                output = file.path(temp_path, 
                                                   'dist_buildings.tif')))
    
    file.remove(file.path(temp_path,
                          glue::glue('temp_{tile_nm}.tif'))); gc()
    
    dists <- do.call(c, lapply(list.files(temp_path, 
                                   pattern = '.tif', 
                                   full.names = T),
                     function(fname) rast(fname)))
    dists <- crop(dists, plts$os_band1)
    unlink(temp_path, recursive = T); gc()
    
    imgs <- c(plts, s1_vv, s1_vh, dists)
    rm(plts, s1_vv, s1_vh, dists); gc()
    imgs
}

get_img <- function(tile_nm, tiles_north,
                          rivers_all, roads_all, 
                          buildings_all){
    message(tile_nm)
    
    # read Planet images
    message('--Planet basemap')
    plt_nms_tile <- grep(tile_nm, plt_nms, value = TRUE)
    plt_os <- rast(grep('2017-12_2018-05', 
                        plt_nms_tile, value = TRUE)) %>% 
        subset(1:4)
    names(plt_os) <- paste0('band', 1:4)
    plt_os$ndvi <- (plt_os$band4 - plt_os$band3) / (plt_os$band4 + plt_os$band3)
    plt_os$evi <- 2.5 * ((plt_os$band4 - plt_os$band3) / 
                             (plt_os$band4 + 2.4 * plt_os$band3 + 1))
    plt_os$savi <- ((plt_os$band4 - plt_os$band3) / 
                        (plt_os$band4 + plt_os$band3 + 1)) * 2
    plt_os$arvi <- (plt_os$band4 - (2 * plt_os$band3) + plt_os$band1) / 
        (plt_os$band4 + (2 * plt_os$band3) + plt_os$band1)
    plt_gs <- rast(grep('2018-06_2018-11', 
                        plt_nms_tile, value = TRUE)) %>% 
        subset(1:4)
    names(plt_gs) <- paste0('band', 1:4)
    plt_gs$ndvi <- (plt_gs$band4 - plt_gs$band3) / (plt_gs$band4 + plt_gs$band3)
    plt_gs$evi <- 2.5 * ((plt_gs$band4 - plt_gs$band3) / 
                             (plt_gs$band4 + 2.4 * plt_gs$band3 + 1))
    plt_gs$savi <- ((plt_gs$band4 - plt_gs$band3) / 
                        (plt_gs$band4 + plt_gs$band3 + 1)) * 2
    plt_gs$arvi <- (plt_gs$band4 - (2 * plt_gs$band3) + plt_gs$band1) / 
        (plt_gs$band4 + (2 * plt_gs$band3) + plt_gs$band1)
    plts <- c(plt_os, plt_gs)
    names(plts) <- c(paste('os', names(plt_os), sep = '_'),
                     paste('gs', names(plt_gs), sep = '_'))
    rm(plt_os, plt_gs); gc()
    
    # Read S1 images
    message('--Sentinel-1 images')
    s1_nms_tile <- grep(tile_nm, s1_nms, value = TRUE)
    s1_vv <- rast(grep('VV', s1_nms_tile, value = TRUE))
    s1_vv <- resample(s1_vv, plts$os_band1)
    names(s1_vv) <- paste0('vv', 1:6)
    s1_vh <- rast(grep('VH', s1_nms_tile, value = TRUE))
    s1_vh <- resample(s1_vh, plts$os_band1)
    names(s1_vh) <- paste0('vh', 1:6)
    
    imgs <- c(plts, s1_vv, s1_vh)
    rm(plts, s1_vv, s1_vh); gc()
    imgs
}

## Make stacks
dir.create('/Volumes/elephant/pred_stack')
## Get vectors
fnames <- list.files(here('data/osm'), full.names = T, pattern = '.geojson')
rivers_all <- st_read(fnames[str_detect(fnames, 'rivers', )])
roads_all <- st_read(fnames[str_detect(fnames, '/roads', )])
buildings_all <- st_read(fnames[str_detect(fnames, 'buildings', )])
gc()

## batch processing
mclapply(unique(samples$tile), 
         function(tile_nm){
    # Get imgs and save out
    imgs <- get_img(tile_nm, tiles_north, 
                    rivers_all, roads_all, 
                    buildings_all)
    writeRaster(imgs, 
                file.path('/Volumes/elephant/pred_stack', 
                          paste0(tile_nm, '.tif')))
})

###########################################
##  Step 4: Extract values for training  ##
###########################################
message('Step 4: Extract values for training')

## Get training
training <- do.call(rbind, 
                    mclapply(unique(samples$tile), 
                             function(tile_nm){
    # Get imgs
    imgs <- rast(file.path('/Volumes/elephant/pred_stack', 
                            paste0(tile_nm, '.tif'))) %>% 
        # Remove the distance layers because the dataset is
        # not perfect, it is risky to be features
        subset(setdiff(names(.), 
                       c('rivers_dist', 
                         'buildings_dist',
                         'roads_dist')))
    
    # get samples
    samples_this <- samples %>% filter(tile == tile_nm) %>% 
        dplyr::select(landcover) %>% as_Spatial() %>% vect() %>% 
        project(., imgs)
    trainings_this <- terra::extract(imgs, samples_this) %>% 
        mutate(landcover = samples_this$landcover) %>% 
        dplyr::select(-ID)
    rm(imgs, samples_this); gc()
    trainings_this
}, mc.cores = 10))
save(training, file = file.path(here('data/north'), 'training.rda'))
