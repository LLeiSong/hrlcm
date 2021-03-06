library(parallel)
library(terra)
library(sf)
library(raster)
library(dplyr)
library(tidyr)

dst_path <- 'data/zone5'

## Get tiles
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
st_write(tiles_zone5, file.path(dst_path, 'tiles_zone5.geojson'))

## Sample global labels
lc_labels <- rast('data/intermid/lc_labels.tif')
n_labels <- freq(lc_labels)
# set NA flag
NAflag(lc_labels) <- 0

# Get balanced samples
set.seed(10)
samples_global <- sampleStratified(raster(lc_labels), 
                            size = min(30000, 
                                       min(n_labels[, 'count'])),
                            na.rm = T, sp = T) %>% 
    st_as_sf() %>% mutate(landcover = sum) %>% select(landcover)


## Sample regional labels
### Get the labels for zone5
tiles_zone5_vect <- vect(tiles_zone5)
lc_labels_zone5 <- mask(crop(lc_labels, 
                             tiles_zone5_vect), 
                        tiles_zone5_vect)
rm(lc_labels, tiles_zone5_vect); gc()

# Tundra is basically a mis-classification around water bodies,
# and urban class makes significant confusion with other classes,
# so we just mask out these two classes for the guessing stage.
n_labels <- freq(lc_labels_zone5)
lc_labels_zone5[lc_labels_zone5 == 7] <- 0
lc_labels_zone5[lc_labels_zone5 == 8] <- 0

# ## Urban
# esaglc <- rast('data/landcovers/esaglc_rcl.tif')
# esaglc[esaglc != 8] <- 0
# esaglc <- resample(esaglc,
#                    lc_labels_zone5, 
#                    method = 'near')
# esaglc <- mask(crop(esaglc, 
#                     tiles_zone5_vect), 
#                tiles_zone5_vect)
# urban_mask <- esaglc
# urban_mask[urban_mask == 0] <- 1
# urban_mask[urban_mask == 8] <- 0
# lc_labels_zone5 <- lc_labels_zone5 * urban_mask + esaglc
writeRaster(lc_labels_zone5, file.path(dst_path, 'lc_labels_zone5.tif'))

### Sampling
# set NA flag
NAflag(lc_labels_zone5) <- 0

# Get balanced samples
n_labels <- freq(lc_labels_zone5)
set.seed(10)
samples <- sampleStratified(raster(lc_labels_zone5), 
                            size = min(40000, 
                                       min(n_labels[, 'count'])),
                            na.rm = T, sp = T) %>% 
    st_as_sf() %>% mutate(landcover = sum) %>% select(landcover)

### Read training data
plt_path <- '/Volumes/elephant/plt_nicfi'
plt_nms <- list.files(plt_path, full.names = T)

samples <- st_join(samples, tiles_zone5)
training <- do.call(rbind, 
                    mclapply(unique(samples$tile), 
                             function(tile_nm){
    message(tile_nm)
    # read images
    plt_nms_tile <- grep(tile_nm, plt_nms, value = TRUE)
    plt_os <- rast(grep('2017-12_2018-05', 
                        plt_nms_tile, value = TRUE)) %>% 
        subset(1:4)
    names(plt_os) <- paste0('band', 1:4)
    plt_os$ndvi <- (plt_os$band4 - plt_os$band3) / (plt_os$band4 + plt_os$band3)
    plt_os$evi <- 2.5 * ((plt_os$band4 - plt_os$band3) / 
                             (plt_os$band4 + 2.4 * plt_os$band3 + 1))
    plt_os$gci <- plt_os$band4 / (plt_os$band2 + 0.01) - 1
    plt_os$savi <- ((plt_os$band4 - plt_os$band3) / 
                        (plt_os$band4 + plt_os$band3 + 1)) * 2
    plt_os$arvi <- (plt_os$band4 - (2 * plt_os$band3) + plt_os$band1) / 
        (plt_os$band4 + (2 * plt_os$band3) + plt_os$band1)
    plt_os$sipi <- (plt_os$band4 - plt_os$band1) / 
        (plt_os$band4 - plt_os$band3 + 0.01)
    plt_gs <- rast(grep('2018-06_2018-11', 
                        plt_nms_tile, value = TRUE)) %>% 
        subset(1:4)
    names(plt_gs) <- paste0('band', 1:4)
    plt_gs$ndvi <- (plt_gs$band4 - plt_gs$band3) / (plt_gs$band4 + plt_gs$band3)
    plt_gs$evi <- 2.5 * ((plt_gs$band4 - plt_gs$band3) / 
                             (plt_gs$band4 + 2.4 * plt_gs$band3 + 1))
    plt_gs$gci <- plt_gs$band4 / (plt_gs$band2 + 0.01) - 1
    plt_gs$savi <- ((plt_gs$band4 - plt_gs$band3) / 
                        (plt_gs$band4 + plt_gs$band3 + 1)) * 2
    plt_gs$arvi <- (plt_gs$band4 - (2 * plt_gs$band3) + plt_gs$band1) / 
        (plt_gs$band4 + (2 * plt_gs$band3) + plt_gs$band1)
    plt_gs$sipi <- (plt_gs$band4 - plt_gs$band1) / 
        (plt_gs$band4 - plt_gs$band3 + 0.01)
    plts <- c(plt_os, plt_gs)
    names(plts) <- c(paste('os', names(plt_os), sep = '_'),
                     paste('gs', names(plt_gs), sep = '_'))
    rm(plt_os, plt_gs); gc()
    
    # get samples
    samples_this <- samples %>% filter(tile == tile_nm) %>% 
        select(landcover) %>% as_Spatial() %>% vect() %>% 
        project(., plts)
    trainings_this <- terra::extract(plts, samples_this) %>% 
        mutate(landcover = samples_this$landcover) %>% 
        select(-ID)
    rm(plts, samples_this); gc()
    trainings_this
}, mc.cores = 4))
save(training, file = file.path(dst_path, 'training.rda'))
training <- training %>% 
    arrange(landcover) %>% 
    mutate(landcover = as.factor(landcover))

## tune RF
library(randomForest)
o <- tuneRF(training[, -ncol(training)],
            training$landcover,
            ntreeTry = 200, 
            sampsize = ceiling(.6 * nrow(training)),
            stepFactor = 2, 
            improve = 0.0001)
o <- data.frame(o) %>% arrange(OOBError)

## Fit the model
set.seed(100)
rf <- randomForest(landcover ~., 
                   data = training,
                   mtry = o$mtry[1],
                   ntree = 1000,
                   sampsize = ceiling(.6 * nrow(training)))
save(rf, file = 'data/zone5/rf.rda')

# Predict
pred_tile <- function(rf, tile_nm){
    # read images
    plt_nms_tile <- grep(tile_nm, plt_nms, value = TRUE)
    plt_os <- rast(grep('2017-12_2018-05', 
                        plt_nms_tile, value = TRUE)) %>% 
        subset(1:4)
    names(plt_os) <- paste0('band', 1:4)
    plt_os$ndvi <- (plt_os$band4 - plt_os$band3) / (plt_os$band4 + plt_os$band3)
    plt_os$evi <- 2.5 * ((plt_os$band4 - plt_os$band3) / 
                             (plt_os$band4 + 2.4 * plt_os$band3 + 1))
    plt_os$gci <- plt_os$band4 / (plt_os$band2 + 0.01) - 1
    plt_os$savi <- ((plt_os$band4 - plt_os$band3) / 
                        (plt_os$band4 + plt_os$band3 + 1)) * 2
    plt_os$arvi <- (plt_os$band4 - (2 * plt_os$band3) + plt_os$band1) / 
        (plt_os$band4 + (2 * plt_os$band3) + plt_os$band1)
    plt_os$sipi <- (plt_os$band4 - plt_os$band1) / 
        (plt_os$band4 - plt_os$band3 + 0.01)
    plt_gs <- rast(grep('2018-06_2018-11', 
                        plt_nms_tile, value = TRUE)) %>% 
        subset(1:4)
    names(plt_gs) <- paste0('band', 1:4)
    plt_gs$ndvi <- (plt_gs$band4 - plt_gs$band3) / (plt_gs$band4 + plt_gs$band3)
    plt_gs$evi <- 2.5 * ((plt_gs$band4 - plt_gs$band3) / 
                             (plt_gs$band4 + 2.4 * plt_gs$band3 + 1))
    plt_gs$gci <- plt_gs$band4 / (plt_gs$band2 + 0.01) - 1
    plt_gs$savi <- ((plt_gs$band4 - plt_gs$band3) / 
                        (plt_gs$band4 + plt_gs$band3 + 1)) * 2
    plt_gs$arvi <- (plt_gs$band4 - (2 * plt_gs$band3) + plt_gs$band1) / 
        (plt_gs$band4 + (2 * plt_gs$band3) + plt_gs$band1)
    plt_gs$sipi <- (plt_gs$band4 - plt_gs$band1) / 
        (plt_gs$band4 - plt_gs$band3 + 0.01)
    plts <- c(plt_os, plt_gs)
    names(plts) <- c(paste('os', names(plt_os), sep = '_'),
                     paste('gs', names(plt_gs), sep = '_'))
    pred <- predict(plts, rf)
    levels(pred) <- levels(training$landcover)
    writeRaster(pred, file.path(dst_path, paste0(tile_nm, '_rf_guess.tif')))
    rm(plt_os, plt_gs, plts, pred); gc()
}

mclapply(unique(tiles_zone5$tile), function(tile_nm){
    pred_tile(rf, tile_nm)
}, mc.cores = 4)
