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

#######################################
##  Step 2: Select validate dataset  ##
#######################################
# Arbitrarily select out validate tiles
## Due to the importance of validation dataset, 
## it should be selected carefully.
## it should be roughly balanced and with high quality
message('Step 2: Select validate dataset')

# Get perfect tiles
tiles_perfect <- st_read(here('results/north/label_check_catalog.geojson')) %>% 
    filter(pass == 'yes') %>% 
    filter(score == 10)

# Make summary of these tiles
tiles_summary <- mclapply(1:nrow(tiles_perfect), function(n){
    tile_row <- tiles_perfect %>% slice(n)
    label_nm <- paste0('refine_', tile_row$tile, 
                       "_", tile_row$index, '.tif')
    label <- rast(here(file.path('results/north/guess_labels',
                                 label_nm)))
    freq(label) %>% data.frame() %>% 
        filter(value != 0) %>%
        dplyr::select(-layer) %>%
        pivot_wider(names_from = value, values_from = count) %>% 
        mutate(tile = tile_row$tile,
               index = tile_row$index) %>% 
        dplyr::select(tile, index, setdiff(names(.), c('tile', 'index')))
}, mc.cores = 6) %>% bind_rows()

# Select out validation dataset
tiles_valid <- lapply(as.character(1:7), function(ind){
    if (ind %in% 6:7){
        set.seed(10)
        tiles_summary %>% 
            dplyr::select(tile, index, all_of(ind)) %>% 
            na.omit() %>% arrange(desc(across(ind))) %>% 
            slice(1:(nrow(.) / 2)) %>% 
            sample_n(400)
    } else{
        tiles_summary %>% 
            dplyr::select(tile, index, all_of(ind)) %>% 
            na.omit() %>% arrange(desc(across(ind))) %>% 
            slice(1:100)
    }
})

tiles_valid_selected <- lapply(tiles_valid, function(each){
    each %>% dplyr::select(tile, index)
}) %>% bind_rows() %>% distinct()

tiles_valid_selected <- lapply(
  unique(tiles_valid_selected$tile),
  function(tile_id) {
    set.seed(10)
    tiles_valid_selected %>%
      filter(tile == tile_id) %>%
      sample_n(min(10, nrow(.)))
  }
) %>%
  bind_rows() %>%
  distinct()

# Summarize the labels
## Class 6 and 7 are relatively small.
## Urban and bareland.
tiles_valid_sum <- tiles_valid %>% 
    reduce(full_join, by = c('tile', 'index')) %>% 
    dplyr::select(-c(tile, index)) %>% 
    colSums(na.rm = T) %>% print()

# Save out
### Relatively pure tiles with high quality and good balance
tiles_validate <- st_read(here('results/north/label_check_catalog.geojson')) %>% 
    merge(., tiles_valid_selected, by = c('tile', 'index')) %>% 
    mutate(double_check = 'yes') %>% 
    dplyr::select(tile, index, pass, score, double_check, comment)
st_write(tiles_validate,
         here('results/north/tiles_validate.geojson'))

#################################
##  Step 3: Get train dataset  ##
#################################
message('Step 3: Get train dataset')

# All other tiles with different levels of quality
tiles_train <- st_read(here('results/north/label_check_catalog.geojson')) %>% 
    filter(pass == 'yes') %>% 
    filter(score > 7) %>% 
    mutate(id = paste0(tile, index)) %>% 
    filter(!id %in% paste0(tiles_validate$tile, tiles_validate$index)) %>% 
    dplyr::select(-id)
st_write(tiles_train,
         here('results/north/tiles_train.geojson'))

#################################
##  Step 4: Generate catalogs  ##
#################################
message('Step 4: Generate catalogs')

# Set path
train_path <- here('results/north/dl_train')
valid_path <- here('results/north/dl_valid')

# Train catalog
message('--Catalog of training')
catalog_train <- tiles_train %>%
  st_drop_geometry() %>%
  dplyr::select(tile, index, score) %>%
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
  dplyr::select(tile_id, tile, index, score, label, img)
write.csv(catalog_train, 
          here('results/north/dl_catalog_train.csv'),
          row.names = F)

# Validate catalog
message('--Catalog of validation')
catalog_valid <- tiles_validate %>%
    st_drop_geometry() %>%
    dplyr::select(tile, index, score) %>%
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
  dplyr::select(tile_id, tile, index, score, label, img)
write.csv(catalog_valid, 
          here('results/north/dl_catalog_valid.csv'),
          row.names = F)

#################################
##  Step 5: Reorganize images  ##
#################################
message('Step 5: Reorganize images')

# Set paths
img_from <- '/Volumes/elephant/pred_stack'
label_from <- here('results/north/guess_labels')
if (!dir.exists(train_path)) dir.create(train_path)
if (!dir.exists(valid_path)) dir.create(valid_path)

# Labels
message('--Prepare labels')
## Train
message('----Labels for train')
copy_to <- catalog_train$label
invisible(mclapply(copy_to, function(x){
    file.copy(
        file.path(
        label_from, 
        paste0('refine_',
               gsub('_label', '', 
                    basename(x)))),
        x)
}, mc.cores = 8))

## Validation
message('----Labels for validation')
copy_to <- catalog_valid$label
invisible(mclapply(copy_to, function(x){
    file.copy(
        file.path(
            label_from, 
            paste0('refine_',
                   gsub('_label', '', 
                        basename(x)))),
        x)
}, mc.cores = 8))

# Satellite images
message('--Prepare satellite images')
# Select features
load(here('data/north/forest_vip.rda'))
var_selected <- data.frame(var = names(forest_vip$fit$variable.importance),
                      imp = forest_vip$fit$variable.importance) %>% 
    filter(str_detect(var, c('band')) | # remove indices
               str_detect(var, c('vv')) |
               str_detect(var, c('vh'))) %>% 
    filter(imp > 1000) # remove less important ones

tiles <- unique(c(catalog_train$tile, catalog_valid$tile))
invisible(lapply(tiles, function(tile_id){
    message(tile_id)
    sat <- rast(
        file.path(
            img_from, paste0(tile_id, '.tif')
            ))
    sat <- subset(sat, var_selected$var)
    
    # train
    train_this_tile <- catalog_train %>% 
        filter(tile == tile_id)
    if (nrow(train_this_tile) > 0){
        invisible(lapply(1:nrow(train_this_tile), function(n){
            msk <- rast(train_this_tile %>% 
                            slice(n) %>% 
                            pull(label))
            imgs <- crop(sat, msk)
            writeRaster(imgs, 
                        train_this_tile %>% 
                            slice(n) %>%
                            pull(img))
        }))
    }
    
    # validation
    valid_this_tile <- catalog_valid %>% 
        filter(tile == tile_id)
    if (nrow(valid_this_tile) > 0){
        invisible(lapply(1:nrow(valid_this_tile), function(n){
            msk <- rast(valid_this_tile %>% 
                            slice(n) %>% 
                            pull(label))
            imgs <- crop(sat, msk)
            writeRaster(imgs, 
                        valid_this_tile %>% 
                            slice(n) %>%
                            pull(img))
        }))
    }
}))
