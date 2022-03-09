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
zone_groups <- list(c(1, 2, 3, 4, 5), 
                    c(1, 2, 3, 4),
                    c(1, 2, 3, 5),
                    c(1, 2, 4, 5),
                    c(1, 3, 4, 5),
                    c(2, 3, 4, 5))
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
