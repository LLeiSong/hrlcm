# Title     : Script to get samples for RF
# Objective : To sample landcover types to run
#             a random forest model.
# Created by: Lei Song
# Created on: 02/24/21

#######################
##  Step 1: Setting  ##
#######################
message("Step1: Setting")

## Load packages
library(here)
library(terra)
library(parallel)
library(sf)
library(dplyr)
library(tidyr)
library(stringr)
library(rgrass7)

#############################################
##  Step 2: Sampling from ensemble labels  ##
#############################################
message("Step 2: Sampling from ensemble labels")

# Read the OSM-masked ensemble labels
lc_labels <- rast(here("data/tanzania/lc_labels_mask.tif"))
n_labels <- freq(lc_labels)
sum(n_labels[, 3]) * 0.01 / nrow(n_labels)

## Use GRASS GIS to sample
gisBase <- "/Applications/GRASS-7.9.app/Contents/Resources"
initGRASS(
  gisBase = gisBase,
  home = tempdir(),
  gisDbase = tempdir(),
  mapset = "PERMANENT",
  location = "lc_types",
  override = TRUE
)
execGRASS("g.proj",
  flags = "c",
  proj4 = "+proj=longlat +datum=WGS84 +no_defs"
)
execGRASS("r.in.gdal",
  flags = c("o", "overwrite"),
  input = here("data/tanzania/lc_labels_mask.tif"),
  band = 1,
  output = "lc_types"
)
execGRASS("g.region", raster = "lc_types")

####################################################################
## Install r.sample.category addon
# execGRASS("g.extension", extension = "r.sample.category")
# execGRASS("g.gisenv", set = "GRASS_ADDON_BASE='~/.grass7/addons'")
# Sys.setenv("GRASS_ADDON_BASE" = '~/.grass7/addons') 
####################################################################

execGRASS("r.sample.category",
  flags = c("overwrite"),
  parameters = list(
    input = "lc_types",
    output = "lc_samples",
    npoints = c(rep(2e5, 4), rep(1e5, 2)),
    random_seed = 10
  )
)

use_sf()
samples_esm <- readVECT("lc_samples") %>%
  mutate(landcover = lc_types) %>%
  dplyr::select(landcover)
save(samples_esm, 
     file = here("data/tanzania/samples_esm.rda"))

#########################################
##  Step 3: Sampling from OSM dataset  ##
#########################################
message("Step 3: Sampling from OSM dataset")

## Get sample numbers
nums <- samples_esm %>%
  st_drop_geometry() %>%
  group_by(landcover) %>%
  summarise(n = n())
save(nums, file = "data/tanzania/nums.rda")

## urban/built-up
message("--Urban/built-up")

tiles <- read_sf(
  here("data/geoms/tiles_nicfi.geojson")) %>% 
  dplyr::select(tile) %>% st_union()
fn <- 'open_buildings_v1_polygons_ne_10m_TZA.csv'
urbans <- st_read(here(file.path('data/open_buildings', fn)),
                  int64_as_string = F,
                  stringsAsFactors = F); rm(fn)
urbans <- urbans %>% 
  mutate(latitude = as.numeric(latitude),
         longitude = as.numeric(longitude),
         area_in_meters = as.numeric(area_in_meters),
         confidence = as.numeric(confidence)) %>% 
  # Buildings with area less than 4 pixels might not be representative
  filter(confidence >= 0.8 & area_in_meters > 10 * 4.8^2) %>% 
  mutate(geometry = st_as_sfc(geometry) %>% 
           st_cast('MULTIPOLYGON')) %>% 
  st_as_sf() %>% st_set_crs(4326) %>% 
  st_make_valid() %>% 
  st_intersection(tiles)
set.seed(103)
urban_samples <- urbans %>%
  mutate(area = st_area(.) %>% 
           units::set_units("km2")) %>%
  arrange(-area) %>%
  slice(1:100000) %>%
  st_centroid() %>%
  st_sf() %>%
  mutate(landcover = 8) %>%
  dplyr::select(landcover)
rm(urbans, tiles)
gc()

## wetlands
message("--Wetland")
num_sample <- 2e5 - 
    (nums %>% filter(landcover == 5) %>% pull(n))

## Might be slow for super large number
set.seed(104)
wetlands_samples <- read_sf(
    here("data/vct_tanzania/wetlands.geojson")) %>%
  st_sample(num_sample, exact = T) %>%
  st_cast("POINT") %>%
  st_sf() %>%
  filter(!st_is_empty(.)) %>%
  mutate(landcover = 5)

# ## Use GRASS GIS
# wetlands <- st_read('data/osm/wetlands.geojson')
# writeVECT(wetlands, 'wetlands', v.in.ogr_flags = 'overwrite')
# execGRASS('v.to.rast', flags = c("overwrite"),
#           parameters = list(input = 'wetlands',
#                             output = 'wetlands',
#                             use = 'val'))
# execGRASS('r.sample.category', flags = c("overwrite"),
#           parameters = list(input = 'wetlands',
#                             output = 'wetlands_samples',
#                             npoints = num_sample,
#                             random_seed = 104))
# use_sf()
# wetlands_samples <- readVECT('wetlands_samples') %>%
#     mutate(landcover = 5) %>%
#     dplyr::select(landcover)
# rm(wetlands); gc()

# water
message("--Water")
## Might be slow for super large number
set.seed(105)
water_samples <- read_sf(
    here("data/vct_tanzania/waterbodies.geojson")) %>%
  st_sample(2e5, exact = T) %>%
  st_cast("POINT") %>%
  st_sf() %>%
  filter(!st_is_empty(.)) %>%
  mutate(landcover = 6)

# ## Use GRASS GIS
# water <- st_read('data/osm/waterbodies.geojson')
# writeVECT(water, 'water', v.in.ogr_flags = 'overwrite')
# execGRASS('v.to.rast', flags = c("overwrite"),
#           parameters = list(input = 'water',
#                             output = 'water',
#                             use = 'val'))
# execGRASS('r.sample.category', flags = c("overwrite"),
#           parameters = list(input = 'water',
#                             output = 'water_samples',
#                             npoints = 2e6,
#                             random_seed = 105))
# use_sf()
# water_samples <- readVECT('water_samples') %>%
#     mutate(landcover = 6) %>%
#     dplyr::select(landcover)
# rm(water); gc()

samples_osm <- rbind(
  urban_samples,
  wetlands_samples,
  water_samples
)
save(samples_osm, 
     file = here("data/tanzania/samples_osm.rda"))
rm(
  urban_samples,
  wetlands_samples,
  water_samples
)
gc()

#####################################
##  Step 4: Merge two sample sets  ##
#####################################
message("Step 4: Merge two sample sets")

samples_all <- rbind(
  samples_esm,
  samples_osm %>% rename(geom = geometry)
)
rm(samples_esm, samples_osm)
gc()

## Check numbers
samples_all %>%
  st_drop_geometry() %>%
  group_by(landcover) %>%
  summarize(n = n())
save(samples_all, 
     file = here("data/tanzania/samples_all.rda"))
st_write(samples_all, 
         here("data/tanzania/samples_all.geojson"))
