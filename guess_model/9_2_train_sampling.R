# Title     : Script of sampling
# Objective : To test the impact of sampling ratio and methods.
# Created by: Lei Song
# Created on: 07/31/21
# alias Rscript='/Library/Frameworks/R.framework/Resources/bin/Rscript'

#######################
##  Step 1: Setting  ##
#######################
message('Step 1: Setting')

## Load packages
library(glue)
library(here)
library(sf)
library(dplyr)
library(stringr)

##########################
##  Step 2: Read files  ##
##########################
tiles <- read_sf(here('data/geoms/tiles_nicfi_north.geojson'))
trains <- read.csv(here('results/north/dl_catalog_train.csv'))
valids <- read.csv(here('results/north/dl_catalog_valid.csv'))

# Check number of tiles
# The ratio is about 0.3, so we start test the sampling ratio from 0.4.
nrow(valids) / nrow(trains)

########################
##  Step 3: Sampling  ##
########################

# Random sampling
message('--Random sampling')
message('----Ratio 0.4')

set.seed(10)
tiles_selected <- tiles %>% 
    sample_frac(0.4)
trains_selected <- trains %>% 
    filter(tile %in% tiles_selected$tile)
write.csv(
    trains_selected, 
    here('results/north/dl_catalog_train_random_04.csv'))

message('----Ratio 0.6')

set.seed(11)
tiles_selected <- tiles %>% 
    sample_frac(0.6)
trains_selected <- trains %>% 
    filter(tile %in% tiles_selected$tile)
write.csv(
    trains_selected, 
    here('results/north/dl_catalog_train_random_06.csv'))

message('----Ratio 0.8')

set.seed(12)
tiles_selected <- tiles %>% 
    sample_frac(0.8)
trains_selected <- trains %>% 
    filter(tile %in% tiles_selected$tile)
write.csv(
    trains_selected, 
    here('results/north/dl_catalog_train_random_08.csv'))

# Use a median ratio 0.6 to have distinguish difference
ids <- strsplit(unique(tiles$tile), "-")
cols <- unique(unlist(lapply(ids, function(id) id[1])))
rows <- unique(unlist(lapply(ids, function(id) id[2])))
tiles <- tiles %>% 
    mutate(ids = tile) %>% 
    separate(ids, c('col', 'row'))

message('--Horizontal sampling')
trains_selected <- do.call(rbind, lapply(rows, function(row_id){
    set.seed(row_id)
    tiles_selected <- tiles %>% 
        filter(row == row_id) %>% 
        sample_frac(0.6)
    trains %>% 
        filter(tile %in% tiles_selected$tile)
}))
write.csv(
    trains_selected, 
    here('results/north/dl_catalog_train_hori_06.csv'))

message('--Vertical sampling')
trains_selected <- do.call(rbind, lapply(cols, function(col_id){
    set.seed(col_id)
    tiles_selected <- tiles %>% 
        filter(col == col_id) %>% 
        sample_frac(0.6)
    trains %>% 
        filter(tile %in% tiles_selected$tile)
}))
write.csv(
    trains_selected, 
    here('results/north/dl_catalog_train_vert_06.csv'))
