# A casual script to subset datasets
library(here)
library(stringr)
library(dplyr)
library(parallel)
src_dir <- 'results/north'
dst_dir <- '/Users/leisong/Downloads/north'
noise_ratio <- 0.3

catalog_train <- read.csv(file.path(src_dir, 'dl_catalog_train.csv'))
catalog_valid <- read.csv(file.path(src_dir, 'dl_catalog_valid.csv'))

sample_perfect <- catalog_train %>% filter(score == 10)
set.seed(10)
catalog_train <- catalog_train %>% filter(score == 9) %>% 
    sample_n(nrow(sample_perfect) * noise_ratio / (1 - noise_ratio))

dir.create(file.path(dst_dir, 'dl_train'), showWarnings = F)
dir.create(file.path(dst_dir, 'dl_valid'), showWarnings = F)
mclapply(1:nrow(catalog_train), function(n){
    ctg_n <- catalog_train %>% 
        slice(n)
    file.copy(here(file.path(src_dir, ctg_n$img)), 
              file.path(dst_dir, ctg_n$img))
    file.copy(here(file.path(src_dir, ctg_n$label)), 
              file.path(dst_dir, ctg_n$label))
}, mc.cores = 6)

write.csv(catalog_train, file.path(dst_dir, 'dl_catalog_train.csv'),
          row.names = F)
write.csv(catalog_valid, file.path(dst_dir, 'dl_catalog_valid.csv'),
          row.names = F)

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
