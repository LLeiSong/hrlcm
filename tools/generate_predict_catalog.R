###############################################################
## Part 1: predict all validate tiles to compare all methods ##
############## Have instant image stack to use ################
###############################################################

# Get tile ids
library(sf)
library(glue)
library(here)
library(dplyr)
library(stringr)
library(terra)
library(parallel)

tile_valid <- read.csv(here('results/north/dl_catalog_valid.csv'),
                       stringsAsFactors = F) %>% 
    pull(tile) %>% unique()

# Prepare satellite image stack for each tile
load(here('data/north/forest_vip.rda'))
var_selected <- data.frame(var = names(forest_vip$fit$variable.importance),
                           imp = forest_vip$fit$variable.importance) %>% 
    filter(str_detect(var, c('band')) | # remove indices
               str_detect(var, c('vv')) |
               str_detect(var, c('vh'))) %>% 
    filter(imp > 1000) # remove less important ones
rm(forest_vip)

img_from <- '/Volumes/elephant/pred_stack'
img_to <- here('results/north/dl_predict')
if (!dir.exists(img_to)) dir.create(img_to)

cp_img <- lapply(tile_valid, function(tile_id){
    message(tile_id)
    sat <- rast(
        file.path(
            img_from, paste0(tile_id, '.tif')
        ))
    sat <- subset(sat, var_selected$var)
    writeRaster(
        sat, 
        file.path(
            img_to, 
            paste0(tile_id, '.tif')))
})

# Generate catalog
dl_catalog_full <- data.frame(tile_id = tile_valid) %>% 
    mutate(img = file.path('dl_predict', 
                           paste0(tile_id, '.tif')))
tiles <- read_sf(
  here('data/geoms/tiles_nicfi_north.geojson'))
dl_catalog_full <- do.call(rbind, lapply(dl_catalog_full$tile_id, function(id){
  tile_this <- tiles %>% filter(tile == id)
  col <- as.integer(unlist(strsplit(tile_this$tile, '-'))[1])
  row <- as.integer(unlist(strsplit(tile_this$tile, '-'))[2])
  tiles_const <- sapply(
    (row + 1):(row - 1), 
    function(row) {paste((col - 1):(col + 1), row, sep = '-')}) %>%
    as.vector()
  tiles_relate <- tiles %>% 
    slice(st_intersects(tile_this, tiles) %>% 
            unlist()) %>% pull(tile)
  tiles_relate <- ifelse(tiles_const %in% tiles_relate, 
                         glue('dl_predict/{tiles_const}.tif'), 'None')
  tiles_relate <- paste(tiles_relate, collapse = ',')
  dl_catalog_full %>% filter(tile_id == id) %>% 
    mutate(tiles_relate = tiles_relate)
}))

write.csv(dl_catalog_full, 
          here('results/north/dl_catalog_predict.csv'),
          row.names = F)

# A test dataset
set.seed(12)
tiles_test <- read.csv(here('results/north/dl_catalog_valid.csv'),
                       stringsAsFactors = F) %>% 
  group_by(hardiness) %>% 
  sample_n(2)
dl_catalog_test <- dl_catalog_full %>% 
  filter(tile_id %in% tiles_test$tile)
write.csv(dl_catalog_test, 
          here('results/north/dl_catalog_test.csv'),
          row.names = F)

################################################
## Part 2: predict other tiles #################
## Need to generate image stack to use #########
################################################
tiles_pred_left <- read_sf(
    here('data/geoms/tiles_nicfi_north.geojson')) %>% 
    filter(!tile %in% dl_catalog_valid_full$tile_id)

# Function to make raster stack
get_img_imp <- function(tile_nm) {
  message(tile_nm)

  # read Planet images
  message("--Planet basemap")
  plt_nms_tile <- grep(tile_nm, plt_nms, value = TRUE)
  plt_os <- rast(grep("2017-12_2018-05",
    plt_nms_tile,
    value = TRUE
  )) %>%
    subset(1:4)
  names(plt_os) <- paste0("band", 1:4)
  plt_gs <- rast(grep("2018-06_2018-11",
    plt_nms_tile,
    value = TRUE
  )) %>%
    subset(1:4)
  names(plt_gs) <- paste0("band", 1:4)
  plts <- c(plt_os, plt_gs)
  names(plts) <- c(
    paste("os", names(plt_os), sep = "_"),
    paste("gs", names(plt_gs), sep = "_")
  )
  rm(plt_os, plt_gs)
  gc()

  # Read S1 images
  message("--Sentinel-1 images")
  s1_nms_tile <- grep(tile_nm, s1_nms, value = TRUE)
  s1_vv <- rast(grep("VV", s1_nms_tile, value = TRUE))
  s1_vv <- subset(s1_vv, c(1, 2, 4))
  s1_vv <- resample(s1_vv, plts$os_band1)
  names(s1_vv) <- paste0("vv", c(1, 2, 4))
  s1_vh <- rast(grep("VH", s1_nms_tile, value = TRUE))
  s1_vh <- subset(s1_vh, c(1, 2, 4))
  s1_vh <- resample(s1_vh, plts$os_band1)
  names(s1_vh) <- paste0("vh", c(1, 2, 4))

  imgs <- c(plts, s1_vv, s1_vh)
  rm(plts, s1_vv, s1_vh)
  gc()
  imgs
}

## Make stacks
dir.create('/Volumes/elephant/pred_stack_left')

## Read file names
plt_path <- '/Volumes/elephant/plt_nicfi'
plt_nms <- list.files(plt_path, full.names = T)
s1_path <- '/Volumes/elephant/sentinel1_hr_coefs'
s1_nms <- list.files(s1_path, full.names = T)

## batch processing
lapply(
  unique(tiles_pred_left$tile),
  function(tile_nm) {
    # Get imgs and save out
    imgs <- get_img_imp(tile_nm)
    writeRaster(
      imgs,
      file.path(
        "/Volumes/elephant/pred_stack_left",
        paste0(tile_nm, ".tif")
      ))
})

# Generate catalog
dl_catalog_pred <- data.frame(
    tile_id = tiles_pred_left$tile) %>% 
    mutate(img = file.path('dl_predict', 
                           paste0(tile_id, '.tif')))
write.csv(dl_catalog_pred, 
          here('results/north/dl_catalog_predict_others.csv'),
          row.names = F)
