##################################################
## Project: Land cover map
## Script purpose: K-mean clustering to cut the study area
## Date: 06/02/2021
## Author: Lei Song
##################################################
library(eifsdm)
library(ClusterR)
library(rgrass7)
library(rmapshaper)
bry <- st_read('data/geoms/tanzania.geojson')
bios <- worldclim2(var = "bio", bry = bry, res = 2.5, 
                   prefix = 'tz',  path = './data', 
                   ifstack = T)
bios_vals <- getValues(bios) %>%
    scale() %>% data.frame() %>%
    mutate(record = ifelse(is.na(wc2.1_2.5m_bio_1), 0, 1))
bios_vals1 <- bios_vals %>% filter(record == 1) %>% 
    dplyr::select(-record)

opt <- Optimal_Clusters_KMeans(bios_vals1, max_clusters = 50, 
                               plot_clusters = T, verbose = F, 
                               criterion = 'distortion_fK', 
                               fK_threshold = 0.95)

km_init <- KMeans_rcpp(bios_vals1, clusters = 22, num_init = 5, 
                       max_iters = 100, initializer = 'kmeans++', 
                       verbose = F)
clusters <- bios_vals$wc2.1_2.5m_bio_1
clusters[!is.na(clusters)] <- km_init$clusters

rst <- bios$wc2.1_2.5m_bio_1
values(rst) <- clusters
writeRaster(rst, 'data/intermid/clusters2.tif')

gisBase <- "/Applications/GRASS-7.9.app/Contents/Resources"
initGRASS(gisBase = gisBase,
          home = tempdir(),
          gisDbase = tempdir(),  
          mapset = 'PERMANENT', 
          location = 'ecozones', 
          override = TRUE)
execGRASS("g.proj", flags = "c", 
          proj4="+proj=longlat +datum=WGS84 +no_defs")
execGRASS('r.in.gdal', flags = c("o", "overwrite"),
          input = here::here('data/intermid/clusters2.tif'),
          band = 1,
          output = "clusters")
execGRASS("g.region", raster = "clusters")
execGRASS("r.to.vect", flags = c('v', 'overwrite'),
          parameters = list(input = "clusters", 
                            output = "clusters",
                            type = "area",
                            column = "value"))
execGRASS("v.clean", flags = c("overwrite"), 
          parameters = list(input = 'clusters', 
                            output = 'clusters_over',
                            error = 'clusters_error',
                            tool = "rmarea",
                            threshold = 1e10))
execGRASS("v.generalize", flags = c("overwrite"), 
          parameters = list(input = 'clusters_over', 
                            output = 'clusters_smooth',
                            error = 'cluster_snake_error',
                            type = "area",
                            method = "snakes",
                            threshold = 2,
                            alpha = 1))
use_sf()
clusters <- readVECT("clusters_smooth") %>% 
    st_make_valid() %>% dplyr::select(-cat) %>% 
    rename(region = value)
st_write(clusters, 'data/intermid/clusters2.geojson')

# We then manually checked the clusters, 
# and merged some small regions, and split the large one.
# This step is inevitable to be manual.
clusters <- st_read('data/intermid/clusters2.geojson') %>% 
    mutate(region = 1:nrow(.))
st_write(clusters, 'data/intermid/clusters_final.geojson')
