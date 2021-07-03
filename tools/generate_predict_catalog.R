###############################################################
## Part 1: predict all validate tiles to compare all methods ##
############## Have instant image stack to use ################
###############################################################

# Get tile ids
library(here)
library(dplyr)
library(stringr)
library(terra)
library(parallel)
tile_valid <- read.csv(here('results/north/dl_catalog_valid.csv'),
                       stringsAsFactors = F) %>% 
    pull(tile) %>% unique()

# Prepare satellite image stack for each tile
load(here('data/north/forest_vip.rda'))
var_selected <- data.frame(var = names(forest_vip$fit$variable.importance),
                           imp = forest_vip$fit$variable.importance) %>% 
    filter(str_detect(var, c('band')) | # remove indices
               str_detect(var, c('vv')) |
               str_detect(var, c('vh'))) %>% 
    filter(imp > 1000) # remove less important ones

img_from <- '/Volumes/elephant/pred_stack'
img_to <- here('results/north/dl_valid_full')
cp_img <- lapply(tile_valid, function(tile_id){
    message(tile_id)
    sat <- rast(
        file.path(
            img_from, paste0(tile_id, '.tif')
        ))
    sat <- subset(sat, var_selected$var)
    writeRaster(
        sat, 
        file.path(
            img_to, 
            paste0(tile_id, '.tif')))
})

# Generate catalog
dl_catalog_pred <- data.frame(tile_id = tile_valid) %>% 
    mutate(img = file.path('dl_valid_full', 
                           paste0(tile_id, '.tif')))
write.csv(dl_catalog_pred, 
          here('results/north/dl_catalog_valid_full.csv'),
          row.names = F)

###################################################
## Part 2: predict all tiles over the study area ##
###### Need to generate image stack to use ########
###################################################
