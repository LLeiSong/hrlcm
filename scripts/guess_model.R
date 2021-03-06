# an experiment
# run RF model on one tile to get the results, and ensemble the RF results.
## get tiles with high diversity
library(parallel)
library(terra)
library(sf)
library(dplyr)
library(tidyr)
select <- dplyr::select
tiles <- st_read('data/geoms/tiles_nicfi.geojson')
ecozones <- st_read('data/intermid/clusters_final.geojson',
                    quiet = T)
tiles <- do.call(rbind, mclapply(1:nrow(ecozones), function(n){
    ecozone <- ecozones %>% slice(n)
    ids <- st_intersects(ecozone, tiles) %>% unlist()
    tiles %>% slice(ids) %>% 
        select(tile) %>% 
        mutate(ecozone = ecozone$region)
}, mc.cores = 12))
tiles <- tiles %>% na.omit()
tiles_zone5 <- tiles %>% filter(ecozone == 5)

# Get diversity
lc_types <- data.frame(id = seq(1, 10),
                       name = c('Cropland', 'Forest', 'Grassland', 
                                'Shrubland', 'Wetland', 'Water', 
                                'Tundra', 'Urban/Built up',
                                'Bareland', 'Snow/Ice'))
lc_types <- lc_types %>% slice(-10)
tiles_zone5_vect <- vect(tiles_zone5)
lc_labels <- rast('data/intermid/lc_labels.tif')
tiles_labels_zone5 <- mclapply(1:nrow(tiles_zone5), function(n){
    labels_each <- crop(lc_labels, tiles_zone5_vect[n, ])
    lb <- freq(labels_each) %>% data.frame() %>% 
        filter(value != 0) %>%
        mutate(count = count / (dim(labels_each)[1] * dim(labels_each)[2])) %>%
        left_join(., lc_types, by = c('value' = 'id')) %>%
        dplyr::select(-c(value, layer)) %>%
        # filter(count >= 0.2) %>% 
        pivot_wider(names_from = name, values_from = count) %>%
        mutate(ratio = rowSums(.),
               diversity = ncol(.),
               evenness = sd(.),
               tile = tiles_zone5_vect[n, ]$tile)
    if (nrow(lb) > 0) lb
}, mc.cores = 8) %>% bind_rows()

tiles_labels_zone5 <- tiles_labels_zone5 %>% 
    filter(!is.na(evenness)) %>% 
    filter(ratio > 0.5) %>% 
    arrange(evenness, desc(diversity), desc(ratio)) %>% 
    slice(1:50)

tiles_labels_zone5 <- do.call(rbind, lapply(1:nrow(tiles_labels_zone5), function(n){
    tb_row <- tiles_labels_zone5 %>% slice(n) %>% 
        select(-c(ratio, diversity, evenness, tile))
    if ((min(tb_row, na.rm = T)) > 0.00057){
        tiles_labels_zone5 %>% slice(n)
    }
}))


tiles_zone5_guess <- tiles_zone5 %>% 
    filter(tile %in% tiles_labels_zone5$tile)
st_write(tiles_zone5, '/Users/pinot/downloads/tiles_zone5.geojson')

tile_nm <- tiles_labels_zone5$tile[4]
plt_path <- '/Volumes/elephant/plt_nicfi'
plt_nms <- list.files(plt_path, full.names = T)
plt_nms_tile <- grep(tile_nm, plt_nms, value = TRUE)

lc_labels <- rast('data/intermid/lc_labels.tif')
lc_labels <- crop(lc_labels, tiles_zone5 %>% 
                           filter(tile == tile_nm) %>% 
                           st_buffer(0.01) %>% 
                           st_sf() %>% vect())
NAflag(lc_labels) <- 0
plt_os <- rast(file.path('data', 
                         basename(grep('2017-12_2018-05', 
                                       plt_nms_tile, 
                                       value = TRUE))))
plt_gs <- rast(file.path('data', 
                         basename(grep('2018-06_2018-11', 
                                       plt_nms_tile, 
                                       value = TRUE))))
lc_labels <- resample(lc_labels, plt_os[[1]], method = 'near')
lc_labels <- mask(lc_labels, plt_os[[1]])
vals <- c(lc_labels, plt_gs, plt_os)
names(vals) <- c('landcover', paste0('grow', 1:7), paste0('off', 1:7))

# random cells
library(raster)
n_labels <- freq(vals[[1]])
set.seed(10)
samples <- sampleStratified(raster(vals[[1]]), 
                            size = min(10000, 
                                       min(n_labels[, 'count'])),
                            na.rm = T)
v <- vals[samples[, 'cell']]
v <- data.frame(v) %>% 
    mutate(landcover = as.factor(landcover))

library(randomForest)
set.seed(10)
rf <- randomForest(
    landcover ~ .,
    data = v,
    mtree = 3000,
    mtry = 8)

pred <- predict(subset(vals, 2:15), rf)
writeRaster(pred, '/Users/pinot/downloads/pred.tif')
pred_prob <- predict(subset(vals, 2:9), rf, type = "prob")

