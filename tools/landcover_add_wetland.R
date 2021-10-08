# Just update wetland because we skip this super minority class in prediction.
# And its spectral or temporal signature is not unique, since wetland could be covered
# by grassland, shrubland, or even big trees.
# We choose to trust our prediction for the rest classes.
library(here)
library(terra)
library(sf)
library(rgrass7)

## set up
gisBase <- '/Applications/GRASS-7.8.app/Contents/Resources'
crs_grass <- crs(pred, proj = T)
initGRASS(gisBase = gisBase,
          home = tempdir(),
          gisDbase = tempdir(),  
          mapset = 'PERMANENT', 
          location = 'pred', 
          override = TRUE)
execGRASS("g.proj", flags = "c", 
          proj4 = crs_grass)
execGRASS('r.in.gdal', flags = c("o", "overwrite"),
          input = lc_pred_path,
          band = 1,
          output = "lc_types")
execGRASS("g.region", raster = "lc_types")

wetland_path <- here('data/osm/wetlands.geojson')
wetlands <- st_read(wetland_path) %>% st_transform(crs = crs_grass)
writeVECT(wetlands, 'wetlands', v.in.ogr_flags = 'overwrite')
execGRASS('v.to.rast', flags = c("overwrite"),
          parameters = list(input = 'wetlands', 
                            output = 'wetlands',
                            use = 'val'))
out_path <- here(file.path('results/dl/prediction',
                           'unet_k15_only_clr_200epc',
                           'wetland_add.tif'))
execGRASS('r.out.gdal', flags = c("overwrite"),
          parameters = list(input = 'wetlands', 
                            output = out_path,
                            nodata = 0,
                            createopt = "COMPRESS=LZW"))
wetlands <- rast(out_path)

# Land cover
lc_pred_path <- here(file.path('results/dl/prediction',
                               'unet_k15_only_clr_200epc',
                               'landcover_north.tif'))
pred <- rast(lc_pred_path)
pred[wetlands == 1] <- 8
out_path <- here(file.path('results/dl/prediction',
                           'unet_k15_only_clr_200epc',
                           'landcover_north_withwl.tif'))
writeRaster(pred, 
            out_path,
            overwrite = T,
            wopt = list(datatype = 'INT1U',
                        gdal=c("COMPRESS=LZW")))

# Confidence
lc_conf_path <- here(file.path('results/dl/prediction',
                               'unet_k15_only_clr_200epc',
                               'landcover_confidence_north.tif'))
conf <- rast(lc_conf_path)
conf[wetlands == 1] <- 100
out_path <- here(file.path('results/dl/prediction',
                           'unet_k15_only_clr_200epc',
                           'landcover_confidence_north_withwl.tif'))
writeRaster(conf,
            out_path,
            overwrite = T,
            wopt = list(datatype = 'INT1U',
                        gdal=c("COMPRESS=LZW")))
