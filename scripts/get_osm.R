# Title     : Script to get OpenStreet dataset
# Objective : Use server download.geofabrik.de 
#             to get OSM data for a huge area.
# Created by: Lei Song
# Created on: 02/24/21

#######################
##  Step 1: Setting  ##
#######################

## Load libraries
library(stringr)
library(parallel)
library(sf)
library(dplyr)
library(tidyr)

## Define the destination folder
dst_path <- 'data/north'

######################################
##  Step 2: Load and download data  ##
######################################
## Get tiles
select <- dplyr::select
tiles_north <- st_read('data/geoms/tiles_nicfi_north.geojson')

## Get OSM
osm_link <- file.path('https://download.geofabrik.de/africa', 
                      'tanzania-latest-free.shp.zip')
download.file(osm_link, 'data/tanzania_osm.zip')
dir.create('data/temp', showWarnings = F)
unzip('data/tanzania_osm.zip', exdir = 'data/temp')

###########################################
##  Step 3: Crop the data to study area  ##
###########################################
## Subset the vectors
tiles_north <- tiles_north %>% st_union()
fnames <- list.files('data/temp', pattern = '.shp', full.names = T)

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
dir.create('data/osm')
st_write(rivers, 'data/osm/rivers.geojson')
st_write(waterbodies, 'data/osm/waterbodies.geojson')
st_write(wetlands, 'data/osm/wetlands.geojson')
st_write(roads, 'data/osm/roads.geojson')
st_write(big_roads, 'data/osm/big_roads.geojson')
st_write(buildings, 'data/osm/buildings.geojson')

## Delete temporary files
unlink("data/temp", recursive = TRUE)
file.remove('data/tanzania_osm.zip')
