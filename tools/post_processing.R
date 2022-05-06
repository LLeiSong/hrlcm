# Post processing
library(here)
library(terra)
library(parallel)
library(sf)
library(dplyr)
library(tidyr)
library(stringr)
library(rgrass7)

###########################################################
########### STEP 1: Fix some problematic tiles ###########
## due to, for example, the failure of S1 harmonic fitting.
## Fix the image and then redo the prediction. 
## For the tiles that the original images are problematic,
## Using manual tile-based classification
## 1246-962 in zone 3, 1234-957, 1229-975, 1205-978 and 1207-971 in zone4
## 1246-962, 1234-957 and 1207-971: s1 harmonic coefs are missing
## 1229-975 and 1205-978: something is wrong with harmonic fitting
## The solution is straightforward, fix the images!
## NOTE: 1205-978 tile need a lower threshold to mask out the background 
## due to the waterbody. Others just had a occasional failure.
###########################################################
# Run `get_img` in `5_get_training.R` to remake the image stack.
tiles_todo <- c('1246-962', '1234-957', '1229-975', '1205-978', '1207-971')
mclapply(tiles_todo,
         function(tile_nm) {
             # Get imgs and save out
             imgs <- get_img(tile_nm)
             writeRaster(
                 imgs,
                 overwrite = TRUE,
                 file.path(
                     "/Volumes/elephant/pred_stack",
                     paste0(tile_nm, ".tif")
                 )
             )
         }, mc.cores = 3)

# Run related lines in `tools/generate_predict_catalog.R` to 
# regenerate tiles for U-Net.
tiles <- c('1246-962', '1234-957', '1229-975', '1205-978', '1207-971')
img_from <- '/Volumes/elephant/pred_stack'
img_to <- here('/Volumes/elephant/predict')
if (!dir.exists(img_to)) dir.create(img_to)

load(here('data/tanzania/forest_vip.rda'))
var_selected <- data.frame(var = names(forest_vip$fit$variable.importance),
                           imp = forest_vip$fit$variable.importance) %>% 
    filter(str_detect(var, c('band')) | # remove indices
               str_detect(var, c('vv')) |
               str_detect(var, c('vh'))) %>% 
    filter(imp > 1000) # remove less important ones
rm(forest_vip)

cp_img <- mclapply(tiles, function(tile_id){
    message(tile_id)
    sat <- rast(
        file.path(
            img_from, paste0(tile_id, '.tif')
        ))
    sat <- subset(sat, var_selected$var)
    writeRaster(
        sat, overwrite = TRUE,
        file.path(
            img_to, 
            paste0(tile_id, '.tif')))
}, mc.cores = 5)

# Rerun U-Net prediction
## Upload files to instance and subset the catalog
predict_catalog <- read.csv('results/tanzania/dl_catalog_predict.csv',
                            stringsAsFactors = FALSE)
tiles <- predict_catalog %>% 
    filter(tile_id %in% tiles) %>% 
    pull(tiles_relate) %>% 
    strsplit(",") %>% 
    unlist() %>% 
    str_extract('[0-9]+-[0-9]+') %>% 
    na.omit() %>% unique()

predict_catalog <- predict_catalog %>% 
    filter(tile_id %in% tiles)
pred_tiles <- unlist(strsplit(predict_catalog$tiles_relate, ','))
pred_tiles <- unique(pred_tiles[pred_tiles != 'None'])
cp_from <- '/Volumes/elephant'
cp_to <- 'ubuntu@*:/home/ubuntu/hrlcm/results/tanzania'
mclapply(rev(pred_tiles), function(fn) {
    cmd <- sprintf('scp  %s/%s %s/%s', 
                   cp_from, fn, cp_to, fn)
    system(cmd, intern = TRUE)
}, mc.cores = 4)
write.csv(predict_catalog, 
          'results/tanzania/dl_catalog_predict_probc.csv',
          row.names = FALSE)

# STEP 2: Fix some artifacts of water surrounding build-up areas
# in zone 3 and zone 4
## Files
fn_zone3 <- list.files('results/tanzania/prediction/zone3/class',
                       full.names = TRUE)
fn_zone4 <- list.files('results/tanzania/prediction/zone4/class',
                       full.names = TRUE)
fn_fixed <- list.files('results/tanzania/prediction/fixed/class',
                       full.names = TRUE)
fns <- c(fn_zone3, fn_zone4, fn_fixed)
rm(fn_zone3, fn_zone4, fn_fixed)

## Check folders
sapply(unique(dirname(fns)), function(dr) {
    dr <- gsub('/class', '/class_fix_step2', dr)
    if (!dir.exists(dr)) dir.create(dr)
})


fix_water_artifact <- function(fn) {
    class <- rast(fn)
    crs_set <- crs(class, proj=TRUE)
    rm(class)
    gisBase <- '/Applications/GRASS-7.8.app/Contents/Resources'
    initGRASS(gisBase = gisBase,
              home = tempdir(),
              gisDbase = tempdir(),  
              mapset = 'PERMANENT', 
              location = 'osm', 
              override = TRUE)
    execGRASS("g.proj", flags = "c", 
              proj4 = crs_set)
    execGRASS('r.in.gdal', flags = c("o", "overwrite"),
              input = fn,
              band = 1,
              output = "lc_types")
    execGRASS("g.region", raster = "lc_types")
    execGRASS('r.null', map = 'lc_types',
              setnull = paste(c(1:5, 7), collapse = ','))
    execGRASS('r.buffer', flags = c("overwrite"),
              parameters = list(input = 'lc_types', 
                                output = 'lc_types_buff',
                                distances = 100))
    execGRASS('r.mapcalc', flags = c("overwrite"),
              expression = 'bup_mask = lc_types_buff > 0')
    execGRASS('r.out.gdal', flags = c("overwrite"),
              parameters = list(input = 'bup_mask', 
                                output = 'results/tanzania/prediction/bup_mask.tif'))
    
    ## mask out the area of build-up.
    class <- rast(fn)
    buf_mask <- rast('results/tanzania/prediction/bup_mask.tif')
    class <- mask(class, buf_mask, 
                  datatype = 'INT1U',
                  gdal=c("COMPRESS=LZW"))
    class <- classify(class, cbind(5, 6), 
                      datatype = 'INT1U',
                      gdal=c("COMPRESS=LZW"))
    class_cover <- rast(fn)
    class <- cover(class, class_cover,
                   datatype = 'INT1U',
                   gdal=c("COMPRESS=LZW"))
    
    writeRaster(class,
                gsub('/class/', '/class_fix_step2/', fn),
                overwrite = T,
                wopt = list(datatype = 'INT1U',
                            gdal=c("COMPRESS=LZW")))
    
    file.remove('results/tanzania/prediction/bup_mask.tif')
    file.remove(list.files(tempdir(), full.names = T))
}

fix_artifacts <- lapply(fns, fix_water_artifact)

# STEP 3: Remove some artifacts of build-up from river basins or waterbody boundaries
## Mainly because of the high reflectance of these area, which is very similar to built-up.
## The method is very similar to step1.

# Get files
fns <- unlist(lapply(c(paste0('zone', 1:5), 'fixed'), function(dr) {
    list.files(sprintf('results/tanzania/prediction/%s/class', dr),
               full.names = TRUE)
}))

# Check directories and modify folder to read from
## Check folders
sapply(unique(dirname(fns)), function(dr) {
    dr <- gsub('/class', '/class_fix_step3', dr)
    if (!dir.exists(dr)) dir.create(dr)
})

## modify
fns <- unname(sapply(fns, function(fn) {
    dr <- gsub('/class', '/class_fix_step2', dirname(fn))
    if (dir.exists(dr)) {
        file.path(dr, basename(fn))
    } else fn
}))

fix_bp_artifacts <- function(fn, rivers, waterbodies) {
    class <- rast(fn)
    crs_set <- crs(class, proj=TRUE)
    rm(class)
    
    gisBase <- '/Applications/GRASS-7.8.app/Contents/Resources'
    initGRASS(gisBase = gisBase,
              home = tempdir(),
              gisDbase = tempdir(),  
              mapset = 'PERMANENT', 
              location = 'osm', 
              override = TRUE)
    execGRASS("g.proj", flags = "c", 
              proj4 = crs_set)
    execGRASS('r.in.gdal', flags = c("o", "overwrite"),
              input = fn,
              band = 1,
              output = "lc_types")
    execGRASS("g.region", raster = "lc_types")
    
    writeVECT(rivers, 'rivers', v.in.ogr_flags = 'overwrite')
    execGRASS('v.to.rast', flags = c("overwrite"),
              parameters = list(input = 'rivers', 
                                output = 'rivers',
                                use = 'val'))
    execGRASS('r.buffer', flags = c("overwrite"),
              parameters = list(input = 'rivers', 
                                output = 'rivers_buff',
                                distances = 20))
    writeVECT(waterbodies, 'waterbodies', v.in.ogr_flags = 'overwrite')
    execGRASS('v.to.rast', flags = c("overwrite"),
              parameters = list(input = 'waterbodies', 
                                output = 'waterbodies',
                                use = 'val'))
    execGRASS('r.buffer', flags = c("overwrite"),
              parameters = list(input = 'waterbodies', 
                                output = 'waterbodies_buff',
                                distances = 20))
    
    execGRASS('r.mapcalc', flags = c("overwrite"),
              expression = 'water_mask = (rivers_buff > 0 ||| waterbodies_buff > 0)')
    
    out_path <- here(file.path('results/tanzania/prediction',
                               'waters.tif'))
    execGRASS('r.out.gdal', flags = c("overwrite"),
              parameters = list(input = 'water_mask', 
                                output = out_path,
                                nodata = 0,
                                createopt = "COMPRESS=LZW"))
    
    ## mask out the area of waters.
    class <- rast(fn)
    buf_mask <- rast(out_path)
    class <- mask(class, buf_mask, 
                  datatype = 'INT1U',
                  gdal=c("COMPRESS=LZW"))
    class <- classify(class, cbind(6, 5),
                      datatype = 'INT1U',
                      gdal=c("COMPRESS=LZW"))
    class_cover <- rast(fn)
    class <- cover(class, class_cover,
                   datatype = 'INT1U',
                   gdal=c("COMPRESS=LZW"))
    
    # if (str_detect(fn, '/class/')) {
    #     dst_path <- gsub('/class/', '/class_fix_step3/', fn)
    # } else if (str_detect(fn, '/class_fix_step2/')) {
    #     dst_path <- gsub('/class_fix_step2/', '/class_fix_step3/', fn)
    # }
    
    dst_path <- gsub('.tif', '_fix_step3.tif', fn)
    
    writeRaster(class,
                dst_path,
                overwrite = T,
                wopt = list(datatype = 'INT1U',
                            gdal=c("COMPRESS=LZW")))
    # Remove temporary file
    file.remove(out_path)
    file.remove(list.files(tempdir(), full.names = T))
    gc()
}

# Read waters
rivers_path <- here('data/vct_tanzania/rivers.geojson')
rivers <- st_read(rivers_path) %>% 
    filter(fclass == 'river') %>% 
    st_transform(crs = 3857)
waterbodies_path <- here('data/vct_tanzania/waterbodies.geojson')
waterbodies <- st_read(waterbodies_path) %>% st_transform(crs = 3857)

fns <- list.files('results/tanzania/prediction', pattern = '.tif',
                  full.names = TRUE)
fix_artifacts <- lapply(fns, function(fn) {
    fix_bp_artifacts(fn, rivers, waterbodies)
    })

# STEP 4: Add waterbodies and wetlands from OSM
add_water_wetland <- function(fn, waterbodies, wetlands) {
    class <- rast(fn)
    gisBase <- '/Applications/GRASS-7.8.app/Contents/Resources'
    crs_grass <- crs(class, proj = T)
    rm(class)
    
    initGRASS(gisBase = gisBase,
              home = tempdir(),
              gisDbase = tempdir(),  
              mapset = 'PERMANENT', 
              location = 'pred', 
              override = TRUE)
    execGRASS("g.proj", flags = "c", 
              proj4 = crs_grass)
    execGRASS('r.in.gdal', flags = c("o", "overwrite"),
              input = fn,
              band = 1,
              output = "lc_types")
    execGRASS("g.region", raster = "lc_types")
    
    writeVECT(waterbodies, 'waterbodies', v.in.ogr_flags = 'overwrite')
    execGRASS('v.to.rast', flags = c("overwrite"),
              parameters = list(input = 'waterbodies', 
                                output = 'waterbodies',
                                use = 'val',
                                value = 5))
    water_path <- here(file.path('results/tanzania/prediction',
                               'waterbody_add.tif'))
    execGRASS('r.out.gdal', flags = c("overwrite"),
              parameters = list(input = 'waterbodies', 
                                output = water_path,
                                nodata = 0,
                                createopt = "COMPRESS=LZW"))
    
    writeVECT(wetlands, 'wetlands', v.in.ogr_flags = 'overwrite')
    execGRASS('v.to.rast', flags = c("overwrite"),
              parameters = list(input = 'wetlands', 
                                output = 'wetlands',
                                use = 'val',
                                value = 8))
    wet_path <- here(file.path('results/tanzania/prediction',
                               'wetland_add.tif'))
    execGRASS('r.out.gdal', flags = c("overwrite"),
              parameters = list(input = 'wetlands', 
                                output = wet_path,
                                nodata = 0,
                                createopt = "COMPRESS=LZW"))
    waterbody <- rast(water_path)
    wetland <- rast(wet_path)
    
    # Land cover
    lc_pred_path <- here(file.path('results/tanzania/prediction',
                                   'class_zone12345.tif'))
    pred <- rast(lc_pred_path)
    pred <- cover(waterbody, pred,
                   datatype = 'INT1U',
                   gdal=c("COMPRESS=LZW"))
    pred <- cover(wetland, pred,
                  datatype = 'INT1U',
                  gdal=c("COMPRESS=LZW"))
    # Set color table
    coltb <- data.frame(red = c(255, 255, 7, 178, 51, 51, 90, 253, 14),
                        blue = c(255, 127, 71, 223, 160, 71, 96, 191, 165),
                        green = c(255, 0, 2, 138, 44, 249, 87, 111, 145))
    
    coltab(pred) <- coltb
    
    # Save out
    out_path <- here(file.path('results/tanzania/prediction',
                               'landcover_final.tif'))
    writeRaster(pred, 
                out_path,
                overwrite = T,
                wopt = list(datatype = 'INT1U',
                            gdal=c("COMPRESS=LZW")))
    
    # Remove temporary files
    file.remove(list.files(tempdir(), full.names = T))
    gc()
}

waterbodies_path <- here('data/vct_tanzania/waterbodies.geojson')
waterbodies <- st_read(waterbodies_path) %>% st_transform(crs = 3857)
# Modify the shape of a large piece of wetland
wetland_path <- here('data/vct_tanzania/wetlands_modify.geojson')
wetlands <- st_read(wetland_path) %>% st_transform(crs = 3857)
add_water_wetland(file.path('results/tanzania/prediction',
                            'class_zone12345.tif'),
                  waterbodies, wetlands)

# STEP OTHERS:
# There could be other post processing steps to apply, e.g., object-based.
