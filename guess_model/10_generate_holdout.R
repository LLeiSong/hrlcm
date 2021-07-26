# Title     : Script to generate validation dataset
# Objective : To randomly select dataset roughly balanced between types
#             for manually check.
# Created by: Lei Song
# Created on: 03/23/21

#######################
##  Step 1: Setting  ##
#######################
message('Step1: Setting')

library(sf)
library(dplyr)
library(here)

################################################
###  Step 2: Sampling for independent check  ###
################################################
message('Step 2: Sampling for independent check')

# Sample
set.seed(10)
samples_check <- read_sf(
    here('data/geoms/tiles_nicfi_north.geojson')) %>% 
    st_sample(rep(2, nrow(.))) %>% 
    st_buffer(5 %>% units::set_units("m")) %>% 
    st_sf() %>% 
    mutate(landcover = '', 
           source = 'humancheck',
           valid = TRUE)

# Convert to use square
samples_check <- do.call(
    rbind, 
    lapply(1:nrow(samples_check), function(n){
        samples_check %>% slice(n) %>% 
            mutate(geometry = st_make_grid(
                geometry, n = 1))
}))

st_write(samples_check, 
         here('data/north/plys_holdout_check.geojson'))

#######################################################
###  Step 3: Add samples from other public datasets ###
#######################################################
# Full polygon field, use centroid as evaluation
samples_mappingafrica <- read_sf(
    here('data/references/user_maps_tz.geojson')) %>% 
    st_make_valid() %>% 
    st_centroid() %>% 
    st_sf() %>% 
    mutate(landcover = 1, source = 'mappingafrica') %>%
    dplyr::select(landcover, source)

# Polygon samples inside of field, use centroid as evaluation
samples_mlhub <- read_sf(
    here('data/references/ref_crops.geojson')) %>% 
    st_make_valid() %>% 
    st_centroid() %>% 
    st_sf() %>% 
    mutate(landcover = 1, source = 'mlhub') %>%
    dplyr::select(landcover, source)
    
samples_pts <- rbind(samples_mappingafrica, samples_mlhub)
rm(samples_mappingafrica, samples_mlhub); gc()
st_write(samples_pts, here('data/north/pts_holdout.geojson'))

# Then the authors check all these small boxes manually.
# In order to have pure samples as possible, the authors might
# move the generated boxes around.