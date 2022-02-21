# Title     : Script to generate catalogs
# Objective : To generate catalogs for training and validation.
# Created by: Lei Song
# Created on: 04/20/21

#######################
##  Step 1: Setting  ##
#######################
message('Step 1: Setting')

## Load packages
library(glue)
library(here)
library(sf)
library(dplyr)
library(stringr)
library(terra)
library(tidyr)
library(purrr)
library(parallel)

##################################################
##  Step 2: Read tiles of train and validation  ##
##################################################
message('Step 2: Read tiles')
tiles_train <- read_sf(here("results/tanzania/catalog_tiles_train.geojson"))
tiles_valid <- read_sf(here("results/tanzania/catalog_tiles_validate.geojson"))

#################################
##  Step 4: Generate catalogs  ##
#################################
message('Step 3: Generate catalogs')

# Set path
# Replace these path based on your own needs
train_path <- 'train'
valid_path <- 'validation'

# Train catalog
message('--Catalog of training')
catalog_train <- tiles_train %>%
  st_drop_geometry() %>%
    filter(use == 1) %>% 
  dplyr::select(tile, index) %>%
  mutate(
    label = file.path(
      train_path,
      paste0(tile, "_", index, "_label.tif")
    ),
    img = file.path(
      train_path,
      paste0(tile, "_", index, "_img.tif")
    ),
    tile_id = paste(tile, index, sep = '_')
  ) %>% 
  dplyr::select(tile_id, tile, index, label, img)
write.csv(catalog_train, 
          here('results/tanzania/dl_catalog_train.csv'),
          row.names = F)

# Validate catalog
message('--Catalog of validation')
catalog_valid <- tiles_valid %>%
    st_drop_geometry() %>%
    filter(use == 1) %>% 
    dplyr::select(tile, index) %>%
  mutate(
    label = file.path(
      valid_path,
      paste0(tile, "_", index, "_label.tif")
    ),
    img = file.path(
      valid_path,
      paste0(tile, "_", index, "_img.tif")
    ),
    tile_id = paste(tile, index, sep = '_')
  ) %>% 
  dplyr::select(tile_id, tile, index, label, img)
write.csv(catalog_valid, 
          here('results/tanzania/dl_catalog_valid.csv'),
          row.names = F)

#################################
##  Step 5: Reorganize images  ##
#################################
message('Step 4: Reorganize images')

# Set paths
train_path <- here('results/tanzania/train')
valid_path <- here('results/tanzania/validation')
img_from <- '/Volumes/elephant/pred_stack'

# # Labels
# message('--Prepare labels')
# ## Train
# message('----Labels for train')
# copy_to <- catalog_train$label
# invisible(mclapply(copy_to, function(x){
#   file.copy(
#     file.path(label_from, basename(x)),
#     file.path(train_path, basename(x)))
# }, mc.cores = 8))
# 
# ## Validation
# message('----Labels for validation')
# copy_to <- catalog_valid$label
# invisible(mclapply(copy_to, function(x){
#   file.copy(
#     file.path(label_from, basename(x)),
#     file.path(valid_path, basename(x)))
# }, mc.cores = 8))

# Satellite images
message('--Prepare satellite images')
# Select features
load(here('data/tanzania/forest_vip.rda'))
var_selected <- data.frame(var = names(forest_vip$fit$variable.importance),
                           imp = forest_vip$fit$variable.importance) %>% 
  filter(str_detect(var, c('band')) | # remove indices
           str_detect(var, c('vv')) |
           str_detect(var, c('vh'))) %>% 
  filter(imp > 1000) # remove less important ones

tiles <- unique(c(catalog_train$tile, catalog_valid$tile))
invisible(mclapply(tiles, function(id){
  message(id)
  sat <- rast(
    file.path(
      img_from, paste0(id, '.tif')
    ))
  sat <- subset(sat, var_selected$var)
  
  # train
  train_this_tile <- catalog_train %>% 
    filter(tile == id)
  if (nrow(train_this_tile) > 0){
    invisible(lapply(1:nrow(train_this_tile), function(n){
      msk <- rast(train_this_tile %>% 
                    slice(n) %>% 
                    pull(label) %>%
                    basename() %>% 
                    file.path(train_path, .))
      imgs <- crop(sat, msk)
      writeRaster(imgs, 
                  train_this_tile %>% 
                    slice(n) %>%
                    pull(img) %>%
                    basename() %>% 
                    file.path(train_path, .))
    }))
  }
  
  # validation
  valid_this_tile <- catalog_valid %>% 
    filter(tile == id)
  if (nrow(valid_this_tile) > 0){
    invisible(lapply(1:nrow(valid_this_tile), function(n){
      msk <- rast(valid_this_tile %>% 
                    slice(n) %>% 
                    pull(label) %>%
                    basename() %>% 
                    file.path(valid_path, .))
      imgs <- crop(sat, msk)
      writeRaster(imgs, 
                  valid_this_tile %>% 
                    slice(n) %>%
                    pull(img) %>%
                    basename() %>% 
                    file.path(valid_path, .))
    }))
  }
}, mc.cores = 12))
