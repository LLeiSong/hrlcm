# Title     : Script to generate catalogs for prediction of label match
# Created by: Lei Song (lsong@ucsb.edu)
# Created on: 10/06/2023

# Load libraries
library(dplyr)
library(stringr)

## Step 1: 2018, the catalog to make prediction of training and validation sub-tiles.
train_catalog <- read.csv("/scratch/lsong36/tanzania/training/dl_catalog_train.csv")
valid_catalog <- read.csv("/scratch/lsong36/tanzania/training/dl_catalog_valid.csv") %>% 
  filter(!tile_id %in% train_catalog$tile_id)
catalog <- rbind(train_catalog, valid_catalog)
write.csv(catalog, '/scratch/lsong36/tanzania/training/dl_catalog_ipredict_2018.csv', row.names = FALSE)

# 2018 to 2019
## Step 2: 2019, the catalog to make concensus labels of training and validation.
train_catalog <- read.csv("/scratch/lsong36/tanzania/training/dl_catalog_train.csv")
valid_catalog <- read.csv("/scratch/lsong36/tanzania/training/dl_catalog_valid.csv") %>% 
  filter(!tile_id %in% train_catalog$tile_id)
catalog <- rbind(train_catalog, valid_catalog)
catalog$label <- gsub('training/train|training/validation', "training/label", catalog$label)
catalog$label <- gsub("_label", "", catalog$label)
catalog$label <- gsub("label/", "label/class_", catalog$label)
catalog$img <- gsub('training', "training_2019", catalog$img)
write.csv(catalog, '/scratch/lsong36/tanzania/training/dl_catalog_ipredict_2019.csv', row.names = FALSE)

# 2018 to 2020
## 2020, the catalog to make concensus labels of training and validation.
train_catalog <- read.csv("/scratch/lsong36/tanzania/training/dl_catalog_train.csv")
valid_catalog <- read.csv("/scratch/lsong36/tanzania/training/dl_catalog_valid.csv") %>% 
  filter(!tile_id %in% train_catalog$tile_id)
catalog <- rbind(train_catalog, valid_catalog)
catalog$label <- gsub('training/train|training/validation', "training/label", catalog$label)
catalog$label <- gsub("_label", "", catalog$label)
catalog$label <- gsub("label/", "label/class_", catalog$label)
catalog$img <- gsub('training', "training_2020", catalog$img)
write.csv(catalog, '/scratch/lsong36/tanzania/training/dl_catalog_ipredict_2020.csv', row.names = FALSE)

# 2018 to 2021
## 2021, the catalog to make concensus labels of training and validation.
train_catalog <- read.csv("/scratch/lsong36/tanzania/training/dl_catalog_train.csv")
valid_catalog <- read.csv("/scratch/lsong36/tanzania/training/dl_catalog_valid.csv") %>% 
  filter(!tile_id %in% train_catalog$tile_id)
catalog <- rbind(train_catalog, valid_catalog)
catalog$label <- gsub('training/train|training/validation', "training/label", catalog$label)
catalog$label <- gsub("_label", "", catalog$label)
catalog$label <- gsub("label/", "label/class_", catalog$label)
catalog$img <- gsub('training', "training_2021", catalog$img)
write.csv(catalog, '/scratch/lsong36/tanzania/training/dl_catalog_ipredict_2021.csv', row.names = FALSE)

# 2018 to 2022
## 2022, the catalog to make concensus labels of training and validation.
train_catalog <- read.csv("/scratch/lsong36/tanzania/training/dl_catalog_train.csv")
valid_catalog <- read.csv("/scratch/lsong36/tanzania/training/dl_catalog_valid.csv") %>% 
  filter(!tile_id %in% train_catalog$tile_id)
catalog <- rbind(train_catalog, valid_catalog)
catalog$label <- gsub('training/train|training/validation', "training/label", catalog$label)
catalog$label <- gsub("_label", "", catalog$label)
catalog$label <- gsub("label/", "label/class_", catalog$label)
catalog$img <- gsub('training', "training_2022", catalog$img)
write.csv(catalog, '/scratch/lsong36/tanzania/training/dl_catalog_ipredict_2022.csv', row.names = FALSE)
