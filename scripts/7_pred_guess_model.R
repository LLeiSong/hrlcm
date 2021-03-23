# Title     : Script to make guess labels
# Objective : To use random forest guesser
#             to get new labels.
# Created by: Lei Song
# Created on: 02/24/21

#######################
##  Step 1: Setting  ##
#######################
message('Step1: Setting')

library(here)
library(terra)
library(dplyr)
library(ramify)
library(ranger)
library(parsnip)
library(tidymodels)
library(stringr)
library(glue)

#############################
###  Step 2: Preparation  ###
#############################
message('Step 2: Preparation')

# Load model
message("--Load model")
load(here('data/north/guess_rf_md.rda'))

# Define function to do prediction
make_pred <- function(img_path,
                      dst_path = 'results/north/prediction'){
    id <- str_extract(img_path, '[0-9]+-[0-9]+')
    message(paste0('--', id))
    imgs <- rast(img_path) %>%
        subset(setdiff(names(.),
                       c('rivers_dist', 'buildings_dist',
                         'roads_dist')))
    
    ## Predict scores and classes
    scores <- predict(imgs, guess_rf_md, type = "prob")
    writeRaster(scores, here(glue('{dst_path}/scores_{id}.tif')))
    pred <- argmax(values(scores))
    classes <- scores[[1]]
    values(classes) <- pred
    writeRaster(classes, here(glue('{dst_path}/classed_{id}.tif')))
}

#############################
#  Step 3: Make prediction  #
#############################
message('Step 3: Make prediction')

# Read image stack
dst_path <- '/Volumes/elephant'
fnames <- list.files(
    file.path(dst_path, 'pred_stack'),
                     full.names = T)
set.seed(456)
fnames <- sample(fnames, size = 200)
lapply(fnames, make_pred)
