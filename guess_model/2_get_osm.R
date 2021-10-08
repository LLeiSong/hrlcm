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
dst_path <- here('data/tanzania')

######################################
##  Step 2: Load and download data  ##
######################################
message('Step 2: Load and download data')

## Get tiles
select <- dplyr::select
tiles_north <- st_read(here('data/geoms/tiles_nicfi.geojson'))

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

# Change to use Open Buildings dataset
### Follow the tutorial to download Google Open Buildings dataset 
## https://sites.research.google/open-buildings/ 
fn <- 'open_buildings_v1_polygons_ne_10m_TZA.csv'
buildings <- st_read(here(file.path('data/open_buildings', fn)),
                     int64_as_string = F,
                     stringsAsFactors = F); rm(fn)
buildings <- buildings %>% 
    mutate(latitude = as.numeric(latitude),
           longitude = as.numeric(longitude),
           area_in_meters = as.numeric(area_in_meters),
           confidence = as.numeric(confidence)) %>% 
    # Buildings with area less than 4 pixels might not be representative
    # filter(confidence >= 0.8 & area_in_meters > 4 * 4.8^2) %>% 
    mutate(geometry = st_as_sfc(geometry) %>% 
               st_cast('MULTIPOLYGON')) %>% 
    st_as_sf() %>% st_set_crs(4326) %>% 
    st_make_valid()

## Save out the processed vectors
dir.create(here('data/vct_tanzania'))
st_write(rivers, here('data/vct_tanzania/rivers.geojson'))
st_write(waterbodies, here('data/vct_tanzania/waterbodies.geojson'))
st_write(wetlands, here('data/vct_tanzania/wetlands.geojson'))
st_write(roads, here('data/vct_tanzania/roads.geojson'))
st_write(big_roads, here('data/vct_tanzania/big_roads.geojson'))
st_write(buildings, here('data/vct_tanzania/buildings.geojson'))

## Delete temporary files
unlink(here("data/temp"), recursive = TRUE)
file.remove(here('data/tanzania_osm.zip'))
