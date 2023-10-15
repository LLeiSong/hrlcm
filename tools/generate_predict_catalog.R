###############################################################
## Part 1: predict all validate tiles to compare all methods ##
############## Have instant image stack to use ################
###############################################################

# Load libraries
library(sf)
library(glue)
library(here)
library(dplyr)
library(stringr)
library(terra)
library(parallel)

# Get tile ids
tiles_full <- read_sf('data/tiles_nicfi.geojson') %>% select(tile)

# Generate catalog for years
yr <- 2022
img_dir <- 'image_cube'
catalog_dir <- sprintf('/scratch/lsong36/tanzania/training_%s', yr)

tiles <- tiles_full
dl_catalog_full <- data.frame(tile_id = tiles$tile) %>% 
  mutate(img = file.path(img_dir, paste0(tile_id, "-", yr, '.tif')))

# Double check if all files exist
fnames <- list.files(file.path("/scratch/lsong36/tanzania", img_dir), 
                     pattern = sprintf("-%s.tif", yr))
fnames <- file.path(img_dir, fnames)
dl_catalog_full <- dl_catalog_full %>% filter(img %in% fnames)

# Remove the missing tile from tile geojson as well.
tiles <- tiles %>% filter(tile %in% dl_catalog_full$tile_id)

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
                         glue('{img_dir}/{tiles_const}-{yr}.tif'), 'None')
  tiles_relate <- paste(tiles_relate, collapse = ',')
  dl_catalog_full %>% filter(tile_id == id) %>% 
    mutate(tiles_relate = tiles_relate)
}))

message(sprintf("Numer of image: %s for year %s.", nrow(dl_catalog_full), yr))

# Save out
write.csv(
  dl_catalog_full, row.names = F,
  file.path(catalog_dir, sprintf('dl_catalog_predict_%s.csv', yr)))
