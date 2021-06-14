#### Just a internal used script to subset datasets for experiments ####
### PART 1: Do the experiment of co-teaching using noisy labels. ###
##        - The same number of training labels, but
##          - 80 % perfect labels + 20 % noisy labels (8 or 9)
##          - 60 % perfect labels + 40 % noisy labels (8 or 9)
### PART 2: Get a random prediction catalog for testing. ###

# Libraries
library(here)
library(stringr)
library(dplyr)
library(parallel)
src_dir <- 'results/north'
dst_dir <- 'results/north'

### PART 1: Do the experiment of co-teaching using noisy labels. ###
catalog_train <- read.csv(file.path(src_dir, 'dl_catalog_train.csv'))
num_train <- (catalog_train %>% filter(score == 10) %>% nrow() %/% 32) * 32

### 80 % perfect labels + 20 % noisy labels (8 or 9) ###
noise_ratio <- 0.2
num_perfect <- round(num_train * (1 - noise_ratio), 0)
num_noisy <- num_train - num_perfect

# Noisy with 9
set.seed(10)
sample_perfect <- catalog_train %>% filter(score == 10) %>% 
    sample_n(num_perfect)
set.seed(10)
catalog_train_9 <- catalog_train %>% filter(score == 9) %>% 
    sample_n(num_noisy) %>% rbind(sample_perfect)
rm(sample_perfect)

write.csv(catalog_train_9, file.path(dst_dir, 'dl_catalog_train_noisy9_20.csv'),
          row.names = F)

# Noisy with 8
set.seed(10)
sample_perfect <- catalog_train %>% filter(score == 10) %>% 
    sample_n(num_perfect)
set.seed(10)
catalog_train_8 <- catalog_train %>% filter(score == 8) %>% 
    sample_n(num_noisy) %>% rbind(sample_perfect)
rm(sample_perfect)

write.csv(catalog_train_8, file.path(dst_dir, 'dl_catalog_train_noisy8_20.csv'),
          row.names = F)

### 60 % perfect labels + 40 % noisy labels (8 or 9) ###
noise_ratio <- 0.4
num_perfect <- round(num_train * (1 - noise_ratio), 0)
num_noisy <- num_train - num_perfect

# Noisy with 9
set.seed(10)
sample_perfect <- catalog_train %>% filter(score == 10) %>% 
    sample_n(num_perfect)
set.seed(10)
catalog_train_9 <- catalog_train %>% filter(score == 9) %>% 
    sample_n(num_noisy) %>% rbind(sample_perfect)
rm(sample_perfect)

write.csv(catalog_train_9, file.path(dst_dir, 'dl_catalog_train_noisy9_40.csv'),
          row.names = F)

# Noisy with 8
set.seed(10)
sample_perfect <- catalog_train %>% filter(score == 10) %>% 
    sample_n(num_perfect)
set.seed(10)
catalog_train_8 <- catalog_train %>% filter(score == 8) %>% 
    sample_n(num_noisy) %>% rbind(sample_perfect)
rm(sample_perfect)

write.csv(catalog_train_8, file.path(dst_dir, 'dl_catalog_train_noisy8_40.csv'),
          row.names = F)

# # Fixed perfect sample numbers
# catalog_train <- catalog_train %>% filter(score == 9) %>% 
#     sample_n(nrow(sample_perfect) * noise_ratio / (1 - noise_ratio))

# # Copy files
# dir.create(file.path(dst_dir, 'dl_train'), showWarnings = F)
# dir.create(file.path(dst_dir, 'dl_valid'), showWarnings = F)
# mclapply(1:nrow(catalog_train), function(n){
#     ctg_n <- catalog_train %>% 
#         slice(n)
#     file.copy(here(file.path(src_dir, ctg_n$img)), 
#               file.path(dst_dir, ctg_n$img))
#     file.copy(here(file.path(src_dir, ctg_n$label)), 
#               file.path(dst_dir, ctg_n$label))
# }, mc.cores = 6)

# Get one prediction
# Select features
load(here('data/north/forest_vip.rda'))
var_selected <- data.frame(var = names(forest_vip$fit$variable.importance),
                           imp = forest_vip$fit$variable.importance) %>% 
    filter(str_detect(var, c('band')) | # remove index
               str_detect(var, c('vv')) |
               str_detect(var, c('vh'))) %>% 
    filter(imp > 1000) # remove less important ones
pred_stack <- rast('/Users/leisong/Dropbox/research/hrlcm/results/north/1224-997.tif')
pred_stack <- subset(pred_stack, var_selected$var)
writeRaster(pred_stack, here('results/north/dl_predict/1224-997.tif'))

catalog_pred <- data.frame(tile_id = '1224-997',
                           img = 'dl_predict/1224-997.tif')
write.csv(catalog_pred, here('results/north/dl_catalog_pred.csv'), row.names = F)
