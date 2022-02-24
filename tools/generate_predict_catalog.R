###############################################################
## Part 1: predict all validate tiles to compare all methods ##
############## Have instant image stack to use ################
###############################################################

# Get tile ids
library(sf)
library(glue)
library(here)
library(dplyr)
library(stringr)
library(terra)
library(parallel)

tiles <- read_sf(here('data/geoms/tiles_nicfi.geojson')) %>% select(tile)

# Prepare satellite image stack for each tile
load(here('data/tanzania/forest_vip.rda'))
var_selected <- data.frame(var = names(forest_vip$fit$variable.importance),
                           imp = forest_vip$fit$variable.importance) %>% 
    filter(str_detect(var, c('band')) | # remove indices
               str_detect(var, c('vv')) |
               str_detect(var, c('vh'))) %>% 
    filter(imp > 1000) # remove less important ones
rm(forest_vip)

img_from <- '/Volumes/elephant/pred_stack'
img_to <- here('/Volumes/elephant/predict')
if (!dir.exists(img_to)) dir.create(img_to)

cp_img <- mclapply(tiles$tile, function(tile_id){
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
}, mc.cores = 11)

# Generate catalog
dl_catalog_full <- data.frame(tile_id = tiles$tile) %>% 
    mutate(img = file.path('predict', 
                           paste0(tile_id, '.tif')))

dl_catalog_full <- do.call(rbind, lapply(dl_catalog_full$tile_id, function(id){
  tile_this <- tiles %>% filter(tile == id)
  col <- as.integer(unlist(strsplit(tile_this$tile, '-'))[1])
  row <- as.integer(unlist(strsplit(tile_this$tile, '-'))[2])
  tiles_const <- sapply(
    (row + 1):(row - 1), 
    function(row) {paste((col - 1):(col + 1), row, sep = '-')}) %>%
    as.vector()
  tiles_relate <- tiles %>% 
    slice(st_intersects(tile_this, tiles) %>% 
            unlist()) %>% pull(tile)
  tiles_relate <- ifelse(tiles_const %in% tiles_relate, 
                         glue('predict/{tiles_const}.tif'), 'None')
  tiles_relate <- paste(tiles_relate, collapse = ',')
  dl_catalog_full %>% filter(tile_id == id) %>% 
    mutate(tiles_relate = tiles_relate)
}))

write.csv(dl_catalog_full, 
          here('results/tanzania/dl_catalog_predict.csv'),
          row.names = F)

# A test dataset
set.seed(123)
tiles_test <- read.csv(here('results/tanzania/dl_catalog_valid.csv'),
                       stringsAsFactors = F) %>% 
  sample_n(15)
dl_catalog_test <- dl_catalog_full %>% 
  filter(tile_id %in% tiles_test$tile)
write.csv(dl_catalog_test, 
          here('results/tanzania/dl_catalog_test.csv'),
          row.names = F)
