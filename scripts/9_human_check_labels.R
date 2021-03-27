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

## Generate catalog
src_dir <- here('results/north/prediction')
fnames <- list.files(here(src_dir), pattern = 'classed')
tile_ids <- str_extract(fnames, '[0-9]+-[0-9]+')
check_catalog <- do.call(rbind, 
                         lapply(tile_ids, 
                                function(tile_id){
    data.frame(index = 1:64,
               tile = tile_id,
               fix = 'no', 
               score = NA) %>% 
        select(tile, index, fix, score)
}))
write.csv(check_catalog, 
          here('results/north/check_catalog.csv'),
          row.names = F)

################################
##  Step 2: Define functions  ##
################################
message('Step 2: Define functions')

## Generate leaflet figure
make_alive_figure <- function(tile_id, tile_index, 
                              plts_dir, vh_dir, label_dir, 
                              check_catalog_path){
    tile_id <- "1214-1002"
    plts_dir <- here('data/plts_nicfi')
    vh_dir <- here('data/s1_harmonic')
    label_dir <- here('results/north/guess_labels')
    check_catalog_path <- here('results/north/check_catalog.csv')
    # Read catalog
    check_catalog <- read.csv(check_catalog_path,
                              stringsAsFactors = F)
    
    # Read planetscope images, os and gs
    prefix <- 'planet_medres_normalized_analytic'
    s1_prefix <- '2017-12_2018-05'
    s2_prefix <- '2018-06_2018-11'
    s1 <- stack(file.path(plts_dir,
                           glue('{prefix}_{s1_prefix}_mosaic_{tile_id}.tif')))
    s2 <- stack(file.path(plts_dir,
                           glue('{prefix}_{s2_prefix}_mosaic_{tile_id}.tif')))
    s1 <- subset(s1, 1:4); s2 <- subset(s2, 1:4)
    vh <- stack(file.path(vh_dir,
                          glue('tile{tile_id}_VH_harmonic.tif')))
    vh <- subset(vh, c(1, 2, 4))
    
    # Generate base maps
    mapviewOptions(raster.size = Inf,
                   mapview.maxpixels = ncell(s1))
    m1 <- viewRGB(s1, 4, 3, 2, map.types = 'Esri.WorldImagery',
                  maxpixels = ncell(s1)/16)
    m2 <- viewRGB(s2, 4, 3, 2, map.types = 'Esri.WorldImagery',
                  maxpixels = ncell(s1)/16)
    m3 <- viewRGB(vh, 2, 3, 1, map.types = 'Esri.WorldImagery',
                  maxpixels = ncell(s1)/16)
    
    # Define tables
    types <- data.frame(label = c('Cropland', 'Forest', 'Grassland',
                                  'Shrubland', 'Water', 'Built-up',
                                  'Bareland'),
                                  ID = 1:7)
    cols <- data.frame(col = c('#ff7f00', '#074702', '#b2df8a',
                               '#33a02c', 'blue', '#5a6057',
                               '#fdbf6f'),
                       ID = 1:7)
    
    # Read labels and set levels
    img_lb <- raster(fnames %>%
                         filter(index == tile_index) %>%
                         pull(path)) %>% ratify()
    levels(img_lb)[[1]] <- merge(levels(img_lb)[[1]], 
                                 types, by = 'ID')
    
    # Make map
    m4 <- mapview(img_lb, 
                  col.region = cols %>% 
                      filter(ID %in% levels(img_lb)[[1]]$ID) %>% 
                      pull(col),
                  map.types = 'Esri.WorldImagery',
                  zoom = 2,
                  alpha.regions = 1,
                  layer.name = 'index1')
    sync(m1, m2, m3, m4)
}

####################################
##  Step 3: Pop tiles one by one  ##
####################################
message('Step 3: Pop tiles one by one')
# Read paths of labels
label_dir <- here('results/north/prediction')
fnames <- list.files(label_dir, 
                     full.names = T,
                     pattern = 'classed') %>% 
    data.frame(path = .) %>% 
    mutate(tile = str_extract(path, '[0-9]+-[0-9]+'))
tiles <- st_read(here('data/geoms/tiles_nicfi_north.geojson')) %>% 
    filter(tile %in% fnames$tile)
