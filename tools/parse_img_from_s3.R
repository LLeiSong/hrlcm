# Get imgs in S3
library(aws.s3)
library(dplyr)
library(parallel)

catalog_pred <- read.csv("results/tanzania/dl_catalog_predict_zone1_1.csv", 
                         stringsAsFactors = FALSE)
fns <- unlist(lapply(catalog_pred$tiles_relate, function(relates) {
    strsplit(relates, ",")[[1]]})) %>% unique()
fns <- fns[fns != "None"]

pth_to <- 'ubuntu@:/home/ubuntu/hrlcm/results/tanzania'
pth_from <- 's3://activemapper/leisong'
mclapply(fns, function(tile_pth) {
    print(tile_pth)
    cmd <- sprintf('aws s3 cp %s/%s %s/%s',
                   pth_from, tile_pth, pth_to, tile_pth)
    system(cmd, intern = TRUE)
}, mc.cores = 10)