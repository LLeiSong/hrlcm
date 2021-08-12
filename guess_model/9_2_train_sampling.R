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
library(tidyverse)

##########################
##  Step 2: Read files  ##
##########################
tiles <- read_sf(here('data/geoms/tiles_nicfi_north.geojson'))
trains <- read.csv(here('results/north/dl_catalog_train.csv'))
valids <- read.csv(here('results/north/dl_catalog_valid.csv'))

##############################
##  Step 3: Sampling ratio  ##
##############################
# Reduce sub tile number within each tile
# Still use all tiles over study area

message('--One each tile')
set.seed(10)
trains_1 <- trains %>% 
    group_by(tile) %>% 
    do(sample_n(., 1)) %>% 
    ungroup()
write.csv(
    trains_1, 
    here('results/north/dl_catalog_train_1.csv'))

message('--Two each tile')
set.seed(11)
trains_2 <- trains %>% 
    group_by(tile) %>% 
    do(sample_n(., 2)) %>% 
    ungroup()
write.csv(
    trains_2, 
    here('results/north/dl_catalog_train_2.csv'))

# Sample tiles, and select 3 sub-tiles from each
# Sampling tiles is a bit faster to generate guess label
# and maybe a bit faster for human checking
# However, might be less representative.
message('--Sampling tiles')

# Comparing to one from each tile
message('----Ratio 0.34')
set.seed(10)
tiles_selected <- tiles %>% 
    sample_frac(0.34)
trains_selected <- trains %>% 
    filter(tile %in% tiles_selected$tile)
write.csv(
    trains_selected, 
    here('results/north/dl_catalog_train_random_34.csv'))

# Comparing to two from each tile
message('----Ratio 0.67')
set.seed(11)
tiles_selected <- tiles %>% 
    sample_frac(0.67)
trains_selected <- trains %>% 
    filter(tile %in% tiles_selected$tile)
write.csv(
    trains_selected, 
    here('results/north/dl_catalog_train_random_67.csv'))

###############################
##  Step 3: Sampling method  ##
###############################
ids <- strsplit(unique(tiles$tile), "-")
cols <- sort(unique(unlist(lapply(ids, function(id) as.integer(id[1])))))
rows <- sort(unique(unlist(lapply(ids, function(id) as.integer(id[2])))))
tiles <- tiles %>% 
    mutate(ids = tile) %>% 
    separate(ids, c('col', 'row')) %>% 
    mutate(col = as.integer(col),
           row = as.integer(row))

message('--Horizontal strip')
tiles_selected <- tiles %>% 
    filter(row %in% rows[!as.logical(seq_len(length(rows)) %% 3)])
trains_selected <- trains %>% 
    filter(tile %in% tiles_selected$tile)
write.csv(
    trains_selected, 
    here('results/north/dl_catalog_train_hori_strip_34.csv'))

tiles_selected <- tiles %>% 
    filter(row %in% rows[as.logical(seq_len(length(rows)) %% 3)])
trains_selected <- trains %>% 
    filter(tile %in% tiles_selected$tile)
write.csv(
    trains_selected, 
    here('results/north/dl_catalog_train_hori_strip_67.csv'))

message('--Vertical strip')
tiles_selected <- tiles %>% 
    filter(col %in% cols[!as.logical(seq_len(length(cols)) %% 3)])
trains_selected <- trains %>% 
    filter(tile %in% tiles_selected$tile)
write.csv(
    trains_selected, 
    here('results/north/dl_catalog_train_vert_strip_34.csv'))

tiles_selected <- tiles %>% 
    filter(col %in% cols[as.logical(seq_len(length(cols)) %% 3)])
trains_selected <- trains %>% 
    filter(tile %in% tiles_selected$tile)
write.csv(
    trains_selected, 
    here('results/north/dl_catalog_train_vert_strip_67.csv'))
