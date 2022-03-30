# Get imgs in S3
library(aws.s3)
library(dplyr)
library(parallel)
library(optparse)
library(here)

# Get command inputs
option_list <- list(
    make_option(c("-z", "--zone"),
                type = "integer", default = 1,
                help = "zone id"),
    make_option(c("-s", "--sub_index"),
                type = "integer", default = NULL,
                help = "the sub index of each zone if it has"))
parms <- parse_args(OptionParser(option_list = option_list))

# Do process
pth <- file.path('/home/ubuntu/hrlcm/results/tanzania',
                 sprintf('/dl_catalog_predict_zone%d_%d.csv',
                         parms$zone, parms$sub_index))
catalog_pred <- read.csv(pth, stringsAsFactors = FALSE)
fns <- unlist(lapply(catalog_pred$tiles_relate, function(relates) {
    strsplit(relates, ",")[[1]]})) %>% unique()
fns <- fns[fns != "None"]

pth_to <- '/home/ubuntu/hrlcm/results/tanzania'
pth_from <- 's3://activemapper/leisong'
invisible(mclapply(fns, function(tile_pth) {
    print(tile_pth)
    cmd <- sprintf('aws s3 cp %s/%s %s/%s',
                   pth_from, tile_pth, pth_to, tile_pth)
    system(cmd, intern = TRUE)
}, mc.cores = 10))