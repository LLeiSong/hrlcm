# Title     : Script to get OpenStreet dataset
# Objective : Use server download.geofabrik.de 
#             to get OSM data for a huge area.
# Created by: Lei Song
# Created on: 02/24/21

#######################
##  Step 1: Setting  ##
#######################
message('Step1: Setting')

## Load libraries
library(here)
library(stringr)
library(parallel)
library(sf)
library(dplyr)
library(tidyr)

## Define the destination folder
dst_path <- here('data/north')

######################################
##  Step 2: Load and download data  ##
######################################
message('Step 2: Load and download data')

## Get tiles
select <- dplyr::select
tiles_north <- st_read(here('data/geoms/tiles_nicfi_north.geojson'))

## Get OSM
osm_link <- file.path('https://download.geofabrik.de/africa', 
                      'tanzania-latest-free.shp.zip')
download.file(osm_link, here('data/tanzania_osm.zip'))
dir.create(here('data/temp'), showWarnings = F)
unzip(here('data/tanzania_osm.zip'), exdir = here('data/temp'))

###########################################
##  Step 3: Crop the data to study area  ##
###########################################
message('Step 3: Crop the data to study area')

## Subset the vectors
tiles_north <- tiles_north %>% st_union()
fnames <- list.files(here('data/temp'), pattern = '.shp', full.names = T)

rivers <- st_read(fnames[str_detect(fnames, 'waterways', )]) %>% 
    st_intersection(tiles_north) %>% 
    st_collection_extract('LINESTRING')
waterbodies <- st_read(fnames[str_detect(fnames, 'water_', )]) %>% 
    st_intersection(tiles_north) %>% 
    filter(fclass %in% c('water', 'reservior')) %>% 
    mutate(area = st_area(.)) %>% 
    filter(area > units::set_units(50000, 'm2')) %>% 
    select(-area)
river_ply <- st_read(fnames[str_detect(fnames, 'water_', )]) %>% 
    st_intersection(tiles_north) %>% 
    filter(fclass == 'river')
waterbodies <- rbind(waterbodies, river_ply) %>% 
    st_collection_extract('POLYGON'); rm(river_ply)

wetlands <- st_read(fnames[str_detect(fnames, 'water_', )]) %>% 
    st_intersection(tiles_north) %>% 
    filter(fclass == 'wetland')

roads <- st_read(fnames[str_detect(fnames, 'roads', )]) %>% 
    st_intersection(tiles_north) %>% 
    select('osm_id', 'code', 'fclass')
railways <- st_read(fnames[str_detect(fnames, 'railways', )]) %>% 
    st_intersection(tiles_north) %>% 
    select('osm_id', 'code', 'fclass')
roads <- rbind(roads, railways) %>% 
    st_collection_extract('LINESTRING')
big_roads <- roads %>% 
    filter(fclass %in% c('primary', 'secondary', 'tertiary'))

buildings <- do.call(rbind, 
                     lapply(fnames[str_detect(fnames, 'buildings', )], 
                            st_read)) %>% 
    st_intersection(tiles_north) %>% 
    st_collection_extract('POLYGON')

## Save out the processed vectors
dir.create(here('data/osm'))
st_write(rivers, here('data/osm/rivers.geojson'))
st_write(waterbodies, here('data/osm/waterbodies.geojson'))
st_write(wetlands, here('data/osm/wetlands.geojson'))
st_write(roads, here('data/osm/roads.geojson'))
st_write(big_roads, here('data/osm/big_roads.geojson'))
st_write(buildings, here('data/osm/buildings.geojson'))

## Delete temporary files
unlink(here("data/temp"), recursive = TRUE)
file.remove(here('data/tanzania_osm.zip'))
