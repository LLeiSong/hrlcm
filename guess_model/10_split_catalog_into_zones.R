# Split catalog into 5 zones
library(sf)
library(dplyr)

# Overlay tiles and agro-ecological zones
tiles <- read_sf('data/geoms/tiles_nicfi.geojson') %>% 
    dplyr::select(tile)
aez <- read_sf('data/geoms/agro_ecological_zone_combined_tanzania.geojson') %>% 
    st_transform(crs = st_crs(tiles))
tiles <- st_join(tiles, aez)

ids <- st_touches(tiles)
ids <- sapply(ids[which(is.na(tiles$zone))], function(x) x[1])

tiles_done <- tiles[which(!is.na(tiles$zone)), ]
tiles_todo <- tiles[which(is.na(tiles$zone)), ]
tiles_todo['zone'] <- st_drop_geometry(tiles)[ids, 'zone']
tiles <- rbind(tiles_done, tiles_todo)
rm(aez, ids, tiles_done, tiles_todo)
write_sf(tiles, 'data/geoms/tiles_nicfi_zones.geojson')

# Split catalog
catalog_train <- read.csv('results/tanzania/dl_catalog_train.csv',
                          stringsAsFactors = FALSE)
catalog_valid <- read.csv('results/tanzania/dl_catalog_valid.csv',
                         stringsAsFactors = FALSE)

# Split into groups
zone_groups <- list(c(1, 2, 4, 5),
                    c(1, 2, 5),
                    c(1, 2),
                    c(2, 5),
                    c(2, 3, 4, 5),
                    c(2, 3, 4),
                    c(2, 3), c(2, 4))
invisible(lapply(zone_groups, function(zone_ids) {
    tile_ids <- tiles %>% filter(zone %in% zone_ids) %>% pull(tile)
    catalog_train_zone <- catalog_train %>% filter(tile %in% tile_ids)
    catalog_valid_zone <- catalog_valid %>% filter(tile %in% tile_ids)
    write.csv(catalog_train_zone, 
              sprintf('results/tanzania/dl_catalog_train_zone%s.csv', 
                      paste0(zone_ids, collapse = "")),
              row.names = FALSE)
    write.csv(catalog_valid_zone, 
              sprintf('results/tanzania/dl_catalog_valid_zone%s.csv', 
                      paste0(zone_ids, collapse = "")),
              row.names = FALSE)
}))

# Split predict tiles
catalog_predict <- read.csv('results/tanzania/dl_catalog_predict.csv',
                            stringsAsFactors = FALSE)
zone_groups <- c(1:5)
invisible(lapply(zone_groups, function(zone_ids) {
    tile_ids <- tiles %>% filter(zone %in% zone_ids) %>% 
        pull(tile) %>% unique()
    catalog_pred_zone <- catalog_predict %>% filter(tile_id %in% tile_ids)
    
    if (zone_ids %in% c(1, 4)) {
        catalog_pred_zone1 <- catalog_pred_zone %>% 
            slice(1:(nrow(.) %/% 2))
        catalog_pred_zone2 <- catalog_pred_zone %>% 
            filter(!tile_id %in% catalog_pred_zone1$tile_id)
        write.csv(catalog_pred_zone1, 
                  sprintf('results/tanzania/dl_catalog_predict_zone%s_1.csv', 
                          paste0(zone_ids, collapse = "")),
                  row.names = FALSE)
        write.csv(catalog_pred_zone2, 
                  sprintf('results/tanzania/dl_catalog_predict_zone%s_2.csv', 
                          paste0(zone_ids, collapse = "")),
                  row.names = FALSE)
    } else {
        write.csv(catalog_pred_zone, 
                  sprintf('results/tanzania/dl_catalog_predict_zone%s.csv', 
                          paste0(zone_ids, collapse = "")),
                  row.names = FALSE)
    }
}))
