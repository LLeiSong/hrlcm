# Title     : Script for human checking
# Objective : To automatically generate leaflet map
#             for human checking the refined guess labels.
# Created by: Lei Song
# Created on: 03/24/21

# Some info
# We tried to generate leaflet map to make alive checking in R,
# but it didn't work fluently. So we decided to use QGIS to check.

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

####################################
##  Step 2: Split tiles into sub  ##
####################################
message('Step 2: Split tiles into sub')

# Read paths of labels
label_dir <- here('results/north/prediction')
fnames <- list.files(label_dir, 
                     full.names = T,
                     pattern = 'classed') %>% 
    data.frame(path = .) %>% 
    mutate(tile = str_extract(path, '[0-9]+-[0-9]+'))
tiles <- st_read(here('data/geoms/tiles_nicfi_north.geojson')) %>% 
    filter(tile %in% fnames$tile)

indices <- matrix(1:64, 8, 8)
indices <- indices[, ncol(indices):1]
tiles <- do.call(rbind, lapply(1:nrow(tiles), function(n){
    tile <- tiles %>% slice(n)
    tiles_grids <- st_make_grid(
        tile, 
        n = c(8, 8)) %>% 
        st_sf() %>% 
        mutate(tile = tile$tile,
               index = as.vector(indices),
               pass = 'yes', score = 0,
               comment = '')
}))
st_write(tiles, here('results/north/label_check_catalog.geojson'))
