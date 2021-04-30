# Title     : Script to ensemble landcover labels
# Objective : To ensemble multiple land cover products to make LC labels.
# Created by: Lei Song
# Created on: 03/22/21

#######################
##  Step 1: Setting  ##
#######################
message('Step1: Setting')

## Load libraries
library(dplyr)
library(terra)
library(sf)
library(here)

############################################
##  Step 2: Mosaic and clip the products  ##
############################################
message('Step 2: Mosaic and clip the products')

# Tiles
tiles <- vect(here('data/geoms/tiles_nicfi_north.geojson'))

# Data directory
ext_dir <- '/Volumes/elephant/landcovers'

# FROM_GLC
message('--FROM_GLC')
fnames <- list.files(file.path(ext_dir, 
                               'FROM_GLC_2017'),
                     full.names = T)
fromglc <- do.call(merge, lapply(fnames, function(fname){
    rast(fname)
})) %>% crop(tiles) %>% mask(tiles)

# GFSAD30AFCE
message('--GFSAD30AFCE')
fnames <- list.files(file.path(ext_dir, 
                               'GFSAD30AFCE_2015'),
                     full.names = T,
                     pattern = 'tif$')
gfsad <- do.call(merge, lapply(fnames, function(fname){
    rast(fname)
})) %>% crop(tiles) %>% mask(tiles)

# ESAGLC
message('--ESAGLC')
fnames <- list.files(file.path(ext_dir, 
                               'ESA_GLC_2018'),
                     full.names = T)
esaglc <- do.call(merge, lapply(fnames, function(fname){
    rast(fname)
})) %>% crop(tiles) %>% mask(tiles)

# Get TANSIS mask
message('--TANSIS')
url_cp <- paste0('https://files.osf.io/v1/resources/',
                 'yrj3h/providers/osfstorage/5c283655',
                 '28af78001a63ee6d?action=download&di',
                 'rect&version=1')
dir.create(here('data/tansis'))
download.file(url_cp, here('data/tansis/cp_pred.zip'))
unzip(here('data/tansis/cp_pred.zip'), exdir = here('data/tansis'))

#########################################
##  Step 3: Redefine land cover types  ##
#########################################
message('Step 3: Redefine land cover types')

# Generate convert tables
message('--Generate convert table')

# LC type used
lc_types <- data.frame(
  id = seq(1, 10),
  name = c(
    "Cropland", "Forest", "Grassland",
    "Shrubland", "Wetland", "Water",
    "Tundra", "Urban/Built up",
    "Bareland", "Snow/Ice"
  )
)

# Based on our check, open forest classes
# are closer to Shrubland class for other products,
names <- c(
  "Closed forest, evergreen needle leaf",
  "Closed forest, evergreen, broad leaf",
  "Closed forest, deciduous needle leaf",
  "Closed forest, deciduous broad leaf",
  "Closed forest, mixed",
  "Closed forest, unknown",
  "Open forest, evergreen needle leaf",
  "Open forest, evergreen broad leaf",
  "Open forest, deciduous needle leaf",
  "Open forest, deciduous broad leaf",
  "Open forest, mixed",
  "Open forest, unknown",
  "Shrubs", "Herbaceous vegetation",
  "Herbaceous wetland", "Moss and lichen",
  "Bare / sparse vegetation",
  "Cultivated and managed vegetation / agriculture (cropland)",
  "Urban / built up",
  "Snow and ice",
  "Permanent water bodies",
  "Open sea"
)
esaglc_table <- data.frame(
  id = c(
    seq(111, 116),
    seq(121, 126),
    20, 30, 90, 100, 60,
    40, 50, 70, 80, 200
  ),
  name = names,
  convert = c(
    rep(2, 6), rep(4, 7),
    3, 5, 7, 9, 1,
    8, 10, 6, 6
  )
)

fromglc_table <- data.frame(
  id = seq(1, 10),
  name = c(
    "Cropland", "Forest", "Grassland",
    "Shrubland", "Wetland", "Water",
    "Tundra", "Impervious surface",
    "Bareland", "Snow/Ice"
  ),
  convert = seq(1, 10)
)

gfsad_table <- data.frame(
  id = seq(0, 2),
  name = c(
    "Water",
    "Non-Cropland",
    "Cropland"
  ),
  convert = c(6, NA, 1)
)

# Convert LC maps
message('--Convert maps')

# Path
dst_path <- here('data/landcovers')
if (!dir.exists(dst_path)) dir.create(dst_path)

# ESA-GLC reclassification
message('----ESAGLC')
esaglc_rclmat <- data.frame(from = esaglc_table$id - 1,
                            to = esaglc_table$id, 
                            becomes = esaglc_table$convert) %>% 
    as.matrix()
esaglc_new <- classify(esaglc, esaglc_rclmat, 
                       right = FALSE, 
                       othersNA = TRUE,
                       filename = file.path(dst_path, 
                                            'esaglc_rcl.tif'),
                       wopt = list(datatype = 'INT1U',
                                   gdal=c("COMPRESS=LZW")))

# FROM-GLC reclassification
message('----ROM-GLC')
fromglc_rclmat <- data.frame(from = fromglc_table$id - 1,
                             to = fromglc_table$id, 
                             becomes = fromglc_table$convert) %>% 
    as.matrix()
fromglc_new <- classify(fromglc, fromglc_rclmat, 
                        right = FALSE, 
                        othersNA = TRUE,
                        filename = file.path(dst_path, 
                                             'fromglc_rcl.tif'),
                        wopt = list(datatype = 'INT1U',
                                    gdal=c("COMPRESS=LZW")))

# GFSAD reclassification
message('----GFSAD')
gfsad_rclmat <- data.frame(from = gfsad_table$id - 1,
                           to = gfsad_table$id, 
                           becomes = gfsad_table$convert) %>% 
    as.matrix()
gfsad_new <- classify(gfsad, gfsad_rclmat, 
                      right = FALSE, 
                      othersNA = TRUE,
                      filename = file.path(dst_path, 'gfsad_rcl.tif'),
                      wopt = list(datatype = 'INT1U',
                                  gdal=c("COMPRESS=LZW")))

###########################
##  Step 4: Make labels  ##
###########################
message('Step 4: Make labels')

# Path
dst_path <- here('data/interim/lc_labels')
if (!dir.exists(dst_path)) dir.create(dst_path)

# Get the existing types
# Because not all types exist in the current study area
ids <- unique(c(unique(esaglc_new), 
                unique(fromglc_new), 
                1)) # For GFSAD and TANSIS
lc_types <- lc_types %>% filter(id %in% ids)
# Even though there is some pixels classified as snow/ice
# this class is too tiny in Tanzania,
# so we delete this class.
lc_types <- lc_types %>% filter(id != 10)

tansis_mask <- rast(list.files(here('data/tansis'), 
                               full.names = T, pattern = '.tif$'))
tansis_mask <- tansis_mask$TZ__7
tansis_mask[is.na(tansis_mask)] <- 0
tansis_mask <- resample(tansis_mask != 0,
                        fromglc_new, 
                        method = 'near')

gfsad_mask <- resample(gfsad_new != 0,
                       fromglc_new, 
                       method = 'near')
crop_mask <- (tansis_mask + gfsad_mask) == 0
lapply(1:nrow(lc_types), function(n){
    lc <- lc_types %>% slice(n)
    message(paste0('--', lc$name))
    if (lc$id == 1){
        message('Just use GFSAD for cropland.')
        lc_lb_emb <- resample(gfsad_new == lc$id, 
                              fromglc_new, 
                              method = 'near')
        gc()
    } else if (lc$id == 6){
        message('Use all three products for water.')
        lc_lb_fromglc <- fromglc_new == lc$id
        lc_lb_esaglc <- resample(esaglc_new == lc$id, 
                                 lc_lb_fromglc, 
                                 method = 'near')
        lc_lb_gfsad <- resample(gfsad_new == lc$id, 
                                lc_lb_fromglc, 
                                method = 'near')
        lc_lb_emb <- (lc_lb_esaglc + 
                          lc_lb_fromglc + 
                          lc_lb_gfsad) >= 3
        lc_lb_emb <- lc_lb_emb * crop_mask
        rm(lc_lb_esaglc, lc_lb_fromglc, lc_lb_gfsad); gc()
    } else {
        message('Not use GFSAD for other types.')
        lc_lb_fromglc <- fromglc_new == lc$id
        lc_lb_esaglc <- resample(esaglc_new == lc$id, 
                                 lc_lb_fromglc,
                                 method = 'near')
        lc_lb_emb <- (lc_lb_esaglc + 
                          lc_lb_fromglc) == 2
        message('Skip the cropland area defined by gfsad and tansis.')
        lc_lb_emb <- lc_lb_emb * crop_mask
        # lc_lb_emb <- lc_lb_emb * gfsad_mask
        rm(lc_lb_esaglc, lc_lb_fromglc); gc()
    }
    
    # Save out
    lc_lb_emb[lc_lb_emb == 1] <- lc$id
    fn <- file.path(dst_path, 
                    paste0(gsub('/| ', '_', 
                                lc$name), 
                           '.tif'))
    writeRaster(lc_lb_emb, fn, 
                wopt = list(datatype = 'INT1U',
                            gdal=c("COMPRESS=LZW")))
    rm(lc_lb_emb); gc()
})

#############################
##  Step 5: Label summary  ##
#############################
message('Step 5: Label summary')
labels_sum <- do.call(rbind, 
                      lapply(1:nrow(lc_types), 
                             function(n){
    lc <- lc_types %>% slice(n)
    message(lc$name)
    fn <- file.path(dst_path, 
                    paste0(gsub('/| ', '_', lc$name), 
                           '.tif'))
    lb_sum <- rast(fn) %>% freq()
    label_count <- 0
    label_frac <- 0
    if (nrow(lb_sum) > 1){
        label_count <- lb_sum[2, 3]
        label_frac <- lb_sum[2, 3] / 
            (lb_sum[1, 3] + lb_sum[2, 3])
    }
    data.frame(name = lc$name, 
               count = label_count,
               frac = label_frac)
}))
labels_sum

# Save out
dst_path <- here('data/interim')
if (!dir.exists(dst_path)) dir.create(dst_path)
write.csv(labels_sum, 
          file.path(dst_path,
                    'labels_sum.csv'),
          row.names = F)

#############################################
##  Step 6: Save out the consensus labels  ##
#############################################
message('Step 6: Save out the consensus labels')

dst_path <- here('data/interim/lc_labels')
labels <- do.call(c, lapply(1:nrow(lc_types), function(n){
    lc <- lc_types %>% slice(n)
    message(lc$name)
    fn <- file.path(dst_path, 
                    paste0(gsub('/| ', '_', lc$name), 
                           '.tif'))
    rast(fn)
}))

# save out
dst_path <- here('data/north')
dir.create(dst_path, showWarnings = F)
fn <- file.path(dst_path, 'lc_labels_north.tif')
writeRaster(labels, fn, overwrite = TRUE,
            wopt = list(datatype = 'INT1U',
                        gdal=c("COMPRESS=LZW")))
