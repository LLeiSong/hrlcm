# Title     : Simplify eco-zone layer
# Objective : To simplify eco-zone, then as a mask.
# Created by: Lei Song
# Created on: 01/29/21

# Load packages
library(here)
library(rgrass7)
library(sf)
library(rmapshaper)

# Remove small areas within the layer
ecozones <- st_read(here('data/geoms/ecozones.geojson'))
gisBase <- '/Applications/GRASS-7.9.app/Contents/MacOS'
initGRASS(gisBase = gisBase,
          home = tempdir(),
          gisDbase = tempdir(),  
          mapset = 'PERMANENT', 
          location = 'ecozones', 
          override = TRUE)
execGRASS("g.proj", flags = "c", 
          proj4="+proj=longlat +datum=WGS84 +no_defs")
writeVECT(ecozones, 'ecozones', v.in.ogr_flags = 'overwrite')
execGRASS("g.region", flags = "overwrite", 
          vector = "ecozones")
execGRASS("v.clean", flags = "overwrite", 
          parameters = list(input = 'ecozones', 
                            output = 'ecozones_clean',
                            error = 'ecozones_error',
                            tool = "rmarea",
                            threshold = 1e10))
# Simplify the shape
use_sf()
ecozones <- readVECT("ecozones_clean") %>% 
    ms_simplify() %>% dplyr::select(-cat)

# Save out
st_write(ecozones, here('data/geoms/ecozones_simple.geojson'))
