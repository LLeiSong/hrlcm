# Define function to do prediction
make_pred <- function(tile_id,
                      sample_tiles,
                      road, # sf object
                      waterbody, # sf object
                      building, # sf object
                      size_sub_tile = 512,
                      skip_class = 8,
                      img_dir,
                      dst_dir = 'results/tanzania/guess_labels'){
    message(paste0('--', tile_id))
    
    # Get image
    imgs <- rast(file.path(img_dir, paste0(tile_id, '.tif')))
    
    # Subset vectors for whole tile
    sub_tiles <- sample_tiles %>% 
        filter(tile == tile_id)
    
    # Loop on sampled sub-tiles to do prediction
    tiles_layout <- aggregate(imgs$os_band1, fact = size_sub_tile)
    values(tiles_layout) <- 1:(4096 / size_sub_tile)^2
    # A bit hard coded for default
    skip_class <- paste0('.pred_', skip_class)
    
    # Cut image
    imgs <- lapply(sub_tiles$index, function(index){
        # Make mask
        mask <- copy(tiles_layout)
        mask <- mask == index
        mask[mask == 0] <- NA
        mask <- disagg(mask, fact = size_sub_tile)
        
        # Subset images
        imgs_sub <- imgs * mask
        imgs_sub <- terra::trim(imgs_sub)
        rm(mask); imgs_sub
    })
    names(imgs) <- sub_tiles$index
    
    # Transform vectors
    crs_mer <- crs(imgs[[1]])
    road <- road %>% 
        st_cast('MULTILINESTRING') %>% 
        st_transform(crs = crs_mer) %>% 
        dplyr::select()
    waterbody <- waterbody %>% 
        st_cast('MULTIPOLYGON') %>% 
        st_transform(crs = crs_mer) %>% 
        dplyr::select()
    building <- building %>% 
        st_cast('MULTIPOLYGON') %>% 
        st_transform(crs = crs_mer) %>% 
        dplyr::select()
    
    # Make prediction and refine the guess labels
    lapply(sub_tiles$index, function(index){
        message(paste0('----', index))
        
        # Do prediction
        message('------Make scores and labels')
        imgs_sub <- imgs[[as.character(index)]]
        scores <- predict(imgs_sub, guess_rf_md, type = "prob", na.rm = T)
        writeRaster(
            scores, 
            here(glue('{dst_dir}/scores_{tile_id}_{index}.tif')),
            overwrite = T)
        scores[[skip_class]] <- 0
        pred <- values(scores); pred[is.na(pred)] <- 0
        pred <- argmax(pred)
        classes <- scores[[1]]
        values(classes) <- pred
        score <- max(scores)
        rm(scores, pred); gc()
        
        # Add vectors
        # Set up GRASS GIS
        message('------Add OSM layers')
        crs_mer <- crs(imgs_sub, proj = T)
        gisBase <- '/Applications/GRASS-7.8.app/Contents/Resources'
        initGRASS(gisBase = gisBase,
                  home = tempdir(),
                  gisDbase = tempdir(),  
                  mapset = 'PERMANENT', 
                  location = glue('segments_{tile_id}_{index}'), 
                  override = TRUE)
        execGRASS("g.proj", flags = "c", 
                  proj4 = crs_mer)
        execGRASS('r.in.gdal', flags = c("o", "overwrite"),
                  input = file.path(
                      dst_dir, 
                      glue('scores_{tile_id}_{index}.tif')),
                  band = 1,
                  output = "score")
        execGRASS("g.region", raster = "score")
        
        use_sf()
        if (nrow(road) > 0){
            writeVECT(road, 'roads', v.in.ogr_flags = 'overwrite')
            execGRASS('v.to.rast', flags = c("overwrite"),
                      parameters = list(input = 'roads', 
                                        output = 'roads',
                                        use = 'val',
                                        value = 3))
            roads_path <- tempfile()
            execGRASS('r.out.gdal', flags = c("m", "overwrite"),
                      output = roads_path,
                      input = "roads")
            fill_roads <- rast(roads_path)
        } else {
            fill_roads <- NULL
        }; rm(road)
        
        if (nrow(waterbody) > 0){
            writeVECT(waterbody, 'waterbodies', v.in.ogr_flags = 'overwrite')
            execGRASS('v.to.rast', flags = c("overwrite"),
                      parameters = list(input = 'waterbodies', 
                                        output = 'waterbodies',
                                        use = 'val',
                                        value = 5))
            waterbody_path <- tempfile()
            execGRASS('r.out.gdal', flags = c("m", "overwrite"),
                      output = waterbody_path,
                      input = "waterbodies")
            fill_waterbodies <- rast(waterbody_path)
        } else {
            fill_waterbodies <- NULL
        }; rm(waterbody)
        
        if (nrow(building) > 0){
            writeVECT(building, 'buildings', v.in.ogr_flags = 'overwrite')
            execGRASS('v.to.rast', flags = c("overwrite"),
                      parameters = list(input = 'buildings', 
                                        output = 'buildings',
                                        use = 'val',
                                        value = 6))
            building_path <- tempfile()
            execGRASS('r.out.gdal', flags = c("m", "overwrite"),
                      output = building_path,
                      input = "buildings")
            fill_buildings <- rast(building_path)
        } else {
            fill_buildings <- NULL
        }; rm(building)
        
        # Gather results
        fills <- compact(list(roads = fill_roads,
                              buildings = fill_buildings,
                              waterbodies = fill_waterbodies))
        rm(fill_roads, fill_buildings, fill_waterbodies)
        
        # Replace values
        if (length(fills) > 0){
            for (i in 1:length(fills)){
                mask_vct <- fills[[i]]
                mask_vct <- is.na(mask_vct)
                classes <- classes * mask_vct
                classes <- cover(classes, fills[[i]], values = 0)
                score[!mask_vct] <- max(values(score))
                rm(mask_vct)
            }
        }
        
        # Save out
        message('------Save out')
        names(classes) <- 'class'
        writeRaster(classes, 
                    file.path(dst_dir, 
                              glue('guess_{tile_id}_{index}.tif')),
                    overwrite = T,
                    wopt = list(datatype = 'INT1U',
                                gdal=c("COMPRESS=LZW")))
        
        names(score) <- 'score'
        writeRaster(
            score, 
            here(glue('{dst_dir}/score_{tile_id}_{index}.tif')),
            overwrite = T)
    })
}

make_pred_noscore <- function(tile_id,
                      sample_tiles,
                      road, # sf object
                      waterbody, # sf object
                      building, # sf object
                      size_sub_tile = 512,
                      skip_class = 8,
                      img_dir,
                      dst_dir = 'results/tanzania/guess_labels_add'){
    message(paste0('--', tile_id))
    
    # Get image
    imgs <- rast(file.path(img_dir, paste0(tile_id, '.tif')))
    
    # Subset vectors for whole tile
    sub_tiles <- sample_tiles %>% 
        filter(tile == tile_id)
    
    # Loop on sampled sub-tiles to do prediction
    tiles_layout <- aggregate(imgs$os_band1, fact = size_sub_tile)
    values(tiles_layout) <- 1:(4096 / size_sub_tile)^2
    # A bit hard coded for default
    skip_class <- paste0('.pred_', skip_class)
    
    # Cut image
    imgs <- lapply(sub_tiles$index, function(index){
        # Make mask
        mask <- copy(tiles_layout)
        mask <- mask == index
        mask[mask == 0] <- NA
        mask <- disagg(mask, fact = size_sub_tile)
        
        # Subset images
        imgs_sub <- imgs * mask
        imgs_sub <- terra::trim(imgs_sub)
        rm(mask); imgs_sub
    })
    names(imgs) <- sub_tiles$index
    
    # Transform vectors
    crs_mer <- crs(imgs[[1]])
    road <- road %>% 
        st_cast('MULTILINESTRING') %>% 
        st_transform(crs = crs_mer) %>% 
        dplyr::select()
    waterbody <- waterbody %>% 
        st_cast('MULTIPOLYGON') %>% 
        st_transform(crs = crs_mer) %>% 
        dplyr::select()
    building <- building %>% 
        st_cast('MULTIPOLYGON') %>% 
        st_transform(crs = crs_mer) %>% 
        dplyr::select()
    
    # Make prediction and refine the guess labels
    lapply(sub_tiles$index, function(index){
        message(paste0('----', index))
        
        # Do prediction
        message('------Make scores and labels')
        imgs_sub <- imgs[[as.character(index)]]
        scores <- predict(imgs_sub, guess_rf_md, type = "prob", na.rm = T)
        writeRaster(
            scores, 
            here(glue('{dst_dir}/scores_{tile_id}_{index}.tif')),
            overwrite = T)
        scores[[skip_class]] <- 0
        pred <- values(scores); pred[is.na(pred)] <- 0
        pred <- argmax(pred)
        classes <- scores[[1]]
        values(classes) <- pred
        rm(scores, pred); gc()
        
        # Add vectors
        # Set up GRASS GIS
        message('------Add OSM layers')
        crs_mer <- crs(imgs_sub, proj = T)
        gisBase <- '/Applications/GRASS-7.8.app/Contents/Resources'
        initGRASS(gisBase = gisBase,
                  home = tempdir(),
                  gisDbase = tempdir(),  
                  mapset = 'PERMANENT', 
                  location = glue('segments_{tile_id}_{index}'), 
                  override = TRUE)
        execGRASS("g.proj", flags = "c", 
                  proj4 = crs_mer)
        execGRASS('r.in.gdal', flags = c("o", "overwrite"),
                  input = file.path(
                      dst_dir, 
                      glue('scores_{tile_id}_{index}.tif')),
                  band = 1,
                  output = "score")
        execGRASS("g.region", raster = "score")
        
        use_sf()
        if (nrow(road) > 0){
            writeVECT(road, 'roads', v.in.ogr_flags = 'overwrite')
            execGRASS('v.to.rast', flags = c("overwrite"),
                      parameters = list(input = 'roads', 
                                        output = 'roads',
                                        use = 'val',
                                        value = 3))
            roads_path <- tempfile()
            execGRASS('r.out.gdal', flags = c("m", "overwrite"),
                      output = roads_path,
                      input = "roads")
            fill_roads <- rast(roads_path)
        } else {
            fill_roads <- NULL
        }; rm(road)
        
        if (nrow(waterbody) > 0){
            writeVECT(waterbody, 'waterbodies', v.in.ogr_flags = 'overwrite')
            execGRASS('v.to.rast', flags = c("overwrite"),
                      parameters = list(input = 'waterbodies', 
                                        output = 'waterbodies',
                                        use = 'val',
                                        value = 5))
            waterbody_path <- tempfile()
            execGRASS('r.out.gdal', flags = c("m", "overwrite"),
                      output = waterbody_path,
                      input = "waterbodies")
            fill_waterbodies <- rast(waterbody_path)
        } else {
            fill_waterbodies <- NULL
        }; rm(waterbody)
        
        if (nrow(building) > 0){
            writeVECT(building, 'buildings', v.in.ogr_flags = 'overwrite')
            execGRASS('v.to.rast', flags = c("overwrite"),
                      parameters = list(input = 'buildings', 
                                        output = 'buildings',
                                        use = 'val',
                                        value = 6))
            building_path <- tempfile()
            execGRASS('r.out.gdal', flags = c("m", "overwrite"),
                      output = building_path,
                      input = "buildings")
            fill_buildings <- rast(building_path)
        } else {
            fill_buildings <- NULL
        }; rm(building)
        
        # Gather results
        fills <- compact(list(roads = fill_roads,
                              buildings = fill_buildings,
                              waterbodies = fill_waterbodies))
        rm(fill_roads, fill_buildings, fill_waterbodies)
        
        # Replace values
        if (length(fills) > 0){
            for (i in 1:length(fills)){
                mask_vct <- fills[[i]]
                mask_vct <- is.na(mask_vct)
                classes <- classes * mask_vct
                classes <- cover(classes, fills[[i]], values = 0)
                rm(mask_vct)
            }
        }
        
        # Save out
        message('------Save out')
        names(classes) <- 'class'
        writeRaster(classes, 
                    file.path(dst_dir, 
                              glue('guess_{tile_id}_{index}.tif')),
                    overwrite = T,
                    wopt = list(datatype = 'INT1U',
                                gdal=c("COMPRESS=LZW")))
    })
}