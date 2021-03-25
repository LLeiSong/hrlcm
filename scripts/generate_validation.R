# Title     : Script to generate validation dataset
# Objective : To randomly select dataset roughly balanced between types
#             for manually check.
# Created by: Lei Song
# Created on: 03/23/21

#######################
##  Step 1: Setting  ##
#######################
message('Step1: Setting')

library(raster)
library(dplyr)
library(here)
library(rgrass7)

#################################
###  Step 2: Generate points  ###
#################################
message('Step 2: Generate points')

## Use GRASS GIS to sample
gisBase <- '/Applications/GRASS-7.8.app/Contents/Resources'
initGRASS(gisBase = gisBase,
          home = tempdir(),
          gisDbase = tempdir(),  
          mapset = 'PERMANENT', 
          location = 'lc_types', 
          override = TRUE)
execGRASS("g.proj", flags = "c", 
          proj4="+proj=longlat +datum=WGS84 +no_defs")
execGRASS('r.in.gdal', flags = c("o", "overwrite"),
          input = here('data/north/lc_labels_north.tif'),
          band = 1,
          output = "lc_types")
execGRASS("g.region", raster = "lc_types")
## Install r.sample.category addon
# execGRASS("g.extension", extension = "r.sample.category")
# execGRASS("g.gisenv", set = "GRASS_ADDON_BASE='~/.grass7/addons'")
# Sys.setenv("GRASS_ADDON_BASE" = '~/.grass7/addons') 
execGRASS('r.sample.category', flags = c("overwrite"),
          parameters = list(input = 'lc_types', 
                            output = 'lc_samples',
                            npoints = c(280, rep(100, 4), 20, rep(100, 3)),
                            random_seed = 10))
use_sf()
samples <- readVECT('lc_samples') %>% 
    dplyr::select(lc_types)
st_write(samples, here('data/north/samples_validation.geojson'))
