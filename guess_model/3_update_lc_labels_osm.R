# Title     : Script to update landcover labels
# Objective : To ensemble data from OSM to update 
#             the ensemble LC labels further.
# Created by: Lei Song
# Created on: 02/24/21

#######################
##  Step 1: Setting  ##
#######################
message('Step1: Setting')

## Load libraries
library(here)
library(terra)
library(parallel)
library(sf)
library(dplyr)
library(tidyr)
library(stringr)
library(rgrass7)

## Define the destination folder
dst_path <- here('data/tanzania')

##################################
##  Step 2: Load and crop data  ##
##################################
message('Step 2: Load and crop data')

## Get tiles
select <- dplyr::select
tiles <- st_read(here('data/geoms/tiles_nicfi.geojson'))

## Get the ensemble labels
lc_labels <- rast(here('data/interim/lc_labels.tif'))
tiles_vect <- vect(tiles)
lc_labels <- mask(crop(lc_labels, tiles_vect), tiles_vect)
rm(tiles_vect); gc()
writeRaster(lc_labels, 
            file.path(dst_path, 'lc_labels.tif'),
            wopt = list(datatype = 'INT1U',
                        gdal=c("COMPRESS=LZW")))

#################################
##  Step 3: Prepare OSM masks  ##
#################################
message('Step 3: Prepare OSM masks')

## Get file names
fnames <- list.files(here('data/vct_tanzania'), full.names = T, 
                     pattern = '.geojson')

## set up
gisBase <- '/Applications/GRASS-7.9.app/Contents/Resources'
initGRASS(gisBase = gisBase,
          home = tempdir(),
          gisDbase = tempdir(),  
          mapset = 'PERMANENT', 
          location = 'osm', 
          override = TRUE)
execGRASS("g.proj", flags = "c", 
          proj4="+proj=longlat +datum=WGS84 +no_defs")
execGRASS('r.in.gdal', flags = c("o", "overwrite"),
          input = file.path(dst_path, 'lc_labels.tif'),
          band = 1,
          output = "lc_types")
execGRASS("g.region", raster = "lc_types")

## OSM mask
### rivers
message('--Rivers')

use_sf()
rivers <- st_read(fnames[str_detect(fnames, 'rivers', )])
writeVECT(rivers, 'rivers', v.in.ogr_flags = 'overwrite')
execGRASS('v.to.rast', flags = c("overwrite"),
          parameters = list(input = 'rivers', 
                            output = 'rivers',
                            use = 'val'))
execGRASS('r.buffer', flags = c("overwrite"),
          parameters = list(input = 'rivers', 
                            output = 'rivers_buff',
                            distances = 30))
execGRASS('r.out.gdal', flags = c("overwrite"),
          parameters = list(input = 'rivers_buff', 
                            output = here('data/vct_tanzania/rivers.tif')))

### Waterbodies
message('--Waterbodies')

waterbodies <- st_read(fnames[str_detect(fnames, 'waterbodies', )])
writeVECT(waterbodies, 'waterbodies', v.in.ogr_flags = 'overwrite')
execGRASS('v.to.rast', flags = c("overwrite"),
          parameters = list(input = 'waterbodies', 
                            output = 'waterbodies',
                            use = 'val'))
execGRASS('r.buffer', flags = c("overwrite"),
          parameters = list(input = 'waterbodies', 
                            output = 'waterbodies_buff',
                            distances = 30))
execGRASS('r.out.gdal', flags = c("overwrite"),
          parameters = list(input = 'waterbodies_buff', 
                            output = here('data/vct_tanzania/waterbodies.tif')))

### roads
message('--Roads')

roads <- st_read(fnames[str_detect(fnames, '/roads', )])
writeVECT(roads, 'roads', v.in.ogr_flags = 'overwrite')
execGRASS('v.to.rast', flags = c("overwrite"),
          parameters = list(input = 'roads', 
                            output = 'roads',
                            use = 'val'))
execGRASS('r.buffer', flags = c("overwrite"),
          parameters = list(input = 'roads', 
                            output = 'roads_buff',
                            distances = 30))
execGRASS('r.out.gdal', flags = c("overwrite"),
          parameters = list(input = 'roads_buff', 
                            output = here('data/vct_tanzania/roads.tif')))

### buildings for urban/built-up
message('--Buildings')

# buildings <- st_read(fnames[str_detect(fnames, 'buildings', )])
# buildings <- st_cast(buildings, "POLYGON")
# ctd_buildings <- st_centroid(buildings)
# ctd_buildings <- ctd_buildings %>% 
#     st_buffer(dist = 0.00025 * 5, endCapStyle = "SQUARE") # save RAM
# ctd_buildings <- ctd_buildings %>% st_union()
# ctd_buildings <- st_sf(ctd_buildings)
# writeVECT(ctd_buildings, 'buildings', v.in.ogr_flags = 'overwrite')
# execGRASS('v.to.rast', flags = c("d", "overwrite"),
#           parameters = list(input = 'buildings', 
#                             output = 'buildings',
#                             use = 'val'))
# execGRASS('r.out.gdal', flags = c("overwrite"),
#           parameters = list(input = 'buildings', 
#                             output = here('data/vct_tanzania/buildings.tif')))

# It would be slow for large dataset, use the python script under 
# the same folder to get buildings.tif

### wetland
message('--Wetlands')

wetlands <- st_read(fnames[str_detect(fnames, 'wetlands', )])
writeVECT(wetlands, 'wetlands', v.in.ogr_flags = 'overwrite')
execGRASS('v.to.rast', flags = c("overwrite"),
          parameters = list(input = 'wetlands', 
                            output = 'wetlands',
                            use = 'val'))
execGRASS('r.buffer', flags = c("overwrite"),
          parameters = list(input = 'wetlands', 
                            output = 'wetlands_buff',
                            distances = 30))
execGRASS('r.out.gdal', flags = c("overwrite"),
          parameters = list(input = 'wetlands_buff', 
                            output = here('data/vct_tanzania/wetlands.tif')))

rm(rivers, waterbodies, roads, 
   ctd_buildings, buildings, wetlands); gc()

###############################
##  Step 4: Apply OSM masks  ##
###############################
message('Step 4: Apply OSM masks')

## Read the mask
fnames <- list.files(here('data/vct_tanzania'), 
                     pattern = '.tif', full.names = T)
masks <- do.call(c, lapply(fnames, function(fname){
    rast(fname)
}))
mask <- sum(masks, na.rm = T); rm(masks); gc()
mask[!is.na(mask)] <- 0
mask[is.na(mask)] <- 1

lc_labels_mask <- lc_labels * mask
lc_labels_mask[lc_labels_mask == 0] <- NA
lc_labels_mask[lc_labels_mask == 6] <- NA # remove water
lc_labels_mask[lc_labels_mask == 8] <- NA # remove urban 
writeRaster(lc_labels_mask, 
            file.path(dst_path, 'lc_labels_mask.tif'),
            wopt = list(datatype = 'INT1U',
                        gdal=c("COMPRESS=LZW")))
