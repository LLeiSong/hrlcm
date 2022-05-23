# A temporary file to split catalog
library(sf)
library(glue)
library(dplyr)
library(parallel)
library(stringr)
## split plys into groups
plys <- st_read('/Volumes/elephant/geoms/tiles_nicfi.geojson')

# Filter the done tiles
tiles_done <- list.files('/Volumes/elephant/sentinel1_hr_coefs') %>%
    data.frame(name = .) %>% 
    mutate(tile = str_extract(name, '[0-9]+-[0-9]+'),
           polar = str_extract(name, '[VH]{2}'))
tiles_vv <- tiles_done %>% filter(polar == 'VV')
tiles_vh <- tiles_done %>% filter(polar == 'VH')
tiles_done <- intersect(tiles_vv$tile, tiles_vh$tile)
plys <- plys %>% filter(!tile %in% tiles_done)

chunkize <- function(x, n) split(x, cut(seq_along(x), n, labels = FALSE)) 
id_groups <- chunkize(1:nrow(plys), 6)

## Read catalog
catalog <- st_read('/Volumes/elephant/catalogs/raw_s1_footprints_tz.geojson')
catalog <- catalog %>% 
    dplyr::select(-no_geom) %>% 
    unique()

## Loop to generate files
plys <- lapply(1:length(id_groups), function(n){
    ply <- plys %>% slice(id_groups[[n]])
    catalog_new <- do.call(rbind, mclapply(1:nrow(ply), function(n){
        ply_each <- ply %>% slice(n)
        catalog_ply <- catalog %>% 
            slice(unique(unlist(st_intersects(ply_each, .)))) %>% 
            mutate(no_geom = n - 1)
    }, mc.cores = 10))
    st_write(catalog_new, glue('/Volumes/elephant/catalogs/s1_ftp_p{n}.geojson'))
    st_write(ply, glue('/Volumes/elephant/geoms/tiles_s1_p{n}.geojson'))
    ply
})
