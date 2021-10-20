# Title     : Script to get training data
# Objective : To extract satellite values
#             from a bunch of images.
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

####################################
##  Step 2: Load related dataset  ##
####################################
message("Step 2: Load related dataset")

## Get samples
samples <- read_sf(
    here("data/tanzania/samples_all.geojson"))
tiles <- read_sf(
    here("data/geoms/tiles_nicfi.geojson")) %>% 
  dplyr::select(tile)
samples <- st_join(samples, tiles)
samples <- samples %>% filter(!is.na(tile))
save(samples, file = here("data/tanzania/samples.rda"))

## Read file names
plt_path <- "/Volumes/elephant/plt_nicfi"
plt_nms <- list.files(plt_path, full.names = T)
s1_path <- "/Volumes/elephant/sentinel1_hr_coefs"
s1_nms <- list.files(s1_path, full.names = T)

####################################
##  Step 3: Prepare image stacks  ##
####################################
message("Step 3: Prepare image stacks")

get_img <- function(tile_nm) {
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
  plt_os$ndvi <- (plt_os$band4 - plt_os$band3) / 
      (plt_os$band4 + plt_os$band3)
  plt_os$evi <- 2.5 * ((plt_os$band4 - plt_os$band3) /
    (plt_os$band4 + 2.4 * plt_os$band3 + 1))
  plt_os$savi <- ((plt_os$band4 - plt_os$band3) /
    (plt_os$band4 + plt_os$band3 + 1)) * 2
  plt_os$arvi <- (plt_os$band4 - (2 * plt_os$band3) + plt_os$band1) /
    (plt_os$band4 + (2 * plt_os$band3) + plt_os$band1)
  plt_gs <- rast(grep("2018-06_2018-11",
    plt_nms_tile,
    value = TRUE
  )) %>%
    subset(1:4)
  names(plt_gs) <- paste0("band", 1:4)
  plt_gs$ndvi <- (plt_gs$band4 - plt_gs$band3) / 
      (plt_gs$band4 + plt_gs$band3)
  plt_gs$evi <- 2.5 * ((plt_gs$band4 - plt_gs$band3) /
    (plt_gs$band4 + 2.4 * plt_gs$band3 + 1))
  plt_gs$savi <- ((plt_gs$band4 - plt_gs$band3) /
    (plt_gs$band4 + plt_gs$band3 + 1)) * 2
  plt_gs$arvi <- (plt_gs$band4 - (2 * plt_gs$band3) + plt_gs$band1) /
    (plt_gs$band4 + (2 * plt_gs$band3) + plt_gs$band1)
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
  s1_vv <- resample(s1_vv, plts$os_band1)
  names(s1_vv) <- paste0("vv", 1:6)
  s1_vh <- rast(grep("VH", s1_nms_tile, value = TRUE))
  s1_vh <- resample(s1_vh, plts$os_band1)
  names(s1_vh) <- paste0("vh", 1:6)

  imgs <- c(plts, s1_vv, s1_vh)
  rm(plts, s1_vv, s1_vh)
  gc()
  imgs
}

## Make stacks
dir.create("/Volumes/elephant/pred_stack")
tiles_exist <- list.files("/Volumes/elephant/pred_stack") %>% 
  str_extract('[0-9]+-[0-9]+')

## batch processing
tiles_todo <- setdiff(unique(tiles$tile), tiles_exist)
mclapply(tiles_todo,
  function(tile_nm) {
    # Get imgs and save out
    imgs <- get_img(tile_nm)
    writeRaster(
      imgs,
      file.path(
        "/Volumes/elephant/pred_stack",
        paste0(tile_nm, ".tif")
      )
    )
  },
  mc.cores = 3
)

# Tile 1210-1018 is problematic, but no big deal.
# It is over deep water area.

###########################################
##  Step 4: Extract values for training  ##
###########################################
message("Step 4: Extract values for training")

## Get training
training <- do.call(
  rbind,
  mclapply(unique(samples$tile),
    function(tile_nm) {
      # Get imgs
      imgs <- rast(file.path(
        "/Volumes/elephant/pred_stack",
        paste0(tile_nm, ".tif")
      ))

      # get samples
      samples_this <- samples %>%
        filter(tile == tile_nm) %>%
        dplyr::select(landcover) %>%
        as_Spatial() %>%
        vect() %>%
        project(., imgs)
      trainings_this <- terra::extract(
        imgs, samples_this) %>%
        mutate(landcover = samples_this$landcover) %>%
        dplyr::select(-ID)
      rm(imgs, samples_this)
      gc()
      trainings_this
    }, mc.cores = 6)
)
save(training, 
     file = file.path(here("data/tanzania"), "training.rda"))
