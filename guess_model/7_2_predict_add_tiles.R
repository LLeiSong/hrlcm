# Title     : Script to make guess labels for some special regions
# Objective : To use random forest guesser
#             to get new labels. At this stage,
#             we used Google Open Buildings as built-up label
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
load(here('data/tanzania/guess_rf_md.rda'))

#############################
#  Step 3: Make prediction  #
#############################
message('Step 3: Make prediction')
source('guess_model/functions.R')

# Directories
message("--Set directories")
stack_dir <- '/Volumes/elephant/pred_stack'
labels_dir <- here('results/tanzania/guess_labels_add')
if (!dir.exists(labels_dir)) dir.create(labels_dir)

# Read vectors
message("--Read vectors")
tiles <- read_sf('data/geoms/tiles_nicfi.geojson') %>% select(tile)
# Subset tiles
oversample_zones <- read_sf('data/geoms/oversample_zones.geojson')
tiles <- tiles %>% slice(unique(unlist(st_intersects(oversample_zones, tiles))))
# Remove existing ones
catalog_exist <- read.csv("results/tanzania/dl_catalog_train.csv")
tiles <- tiles %>% filter(!tile %in% unique(catalog_exist$tile))
rm(catalog_exist)

## Remove the problematic tile
tiles <- tiles %>% filter(tile != '1210-1018')

# Cut the tiles and make 3 samples
# 3 just for train
message('--Generate catalog')
n_sample <- 1
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
               index = as.vector(indices)) %>% 
        slice(sample(1:nrow(.), n_sample))
})) %>% mutate(use = 1, modify = 0, comment = NA)
# st_write(sample_tiles, here('results/tanzania/catalog_sample_tiles_add.geojson'))

# Roads
## Crop and join with tiles first
roads <- read_sf(here('data/vct_tanzania/roads.geojson'))
roads <- roads %>%
    filter(fclass %in% c('primary', 'secondary', 'tertiary',
                         'trunk', 'primary_link', 'secondary_link',
                         'tertiary_link', 'rail'))
roads <- st_join(
    roads, 
    sample_tiles %>% dplyr::select(tile, index)) %>% 
    filter(!is.na(tile))

# Waterbodies
## Crop and join first
waterbodies <- read_sf(here('data/vct_tanzania/waterbodies.geojson'))
waterbodies <- st_join(
    waterbodies, sample_tiles %>% dplyr::select(tile, index)) %>% 
    filter(!is.na(tile))

# Buildings
ranges <- st_bbox(oversample_zones)
fn <- 'open_buildings_v1_polygons_ne_10m_TZA.csv'
buildings <- st_read(here(file.path('data/open_buildings', fn)),
                     int64_as_string = F,
                     stringsAsFactors = F); rm(fn)
buildings <- buildings %>% 
    mutate(latitude = as.numeric(latitude),
           longitude = as.numeric(longitude),
           area_in_meters = as.numeric(area_in_meters),
           confidence = as.numeric(confidence)) %>% 
    filter(latitude >= ranges[2] & latitude <= ranges[4] &
               longitude >= ranges[1] & longitude <= ranges[3]) %>% 
    # Buildings with area less than a pixel might not be representative
    filter(confidence >= 0.75 & area_in_meters > 4.8^2) %>% 
    mutate(geometry = st_as_sfc(geometry) %>% 
               st_cast('MULTIPOLYGON')) %>% 
    st_as_sf() %>% st_set_crs(4326) %>% 
    st_make_valid()
buildings <- st_join(
    buildings, sample_tiles %>% dplyr::select(tile, index)) %>% 
    filter(!is.na(tile))
# save(buildings, file = 'results/tanzania/buildings.rda')

message('--Start generating')
lapply(unique(sample_tiles$tile), function(tile_id){
    road <- roads %>% 
        filter(tile == tile_id)
    waterbody <- waterbodies %>% 
        filter(tile == tile_id)
    building <- buildings %>% 
        filter(tile == tile_id)
    make_pred_noscore(tile_id = tile_id,
                      sample_tiles = sample_tiles,
                      road = road,
                      waterbody = waterbody,
                      building = building,
                      img_dir = stack_dir)
    
    # Remove big spat temps
    file.remove(list.files(tempdir(), pattern = 'spat',
                           full.names = T))
})
