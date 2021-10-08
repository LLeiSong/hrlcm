#!/usr/bin/env sh
# Authors: Lei Song
## Mosaic the final tiled land cover maps, just GTiff

# Get input parameters
SRC=$1
if [ -z "$SRC" ]; then
    echo "`date`: Usage: $0 <src_dir> <mosaic_path>"
    exit 1
fi
DST=$2
if [ -z "$DST" ]; then
    echo "`date`: Usage: $0 <src_dir> <mosaic_path>"
    exit 1
fi

# Do mosaic
echo "Start to mosaic files in $SRC to $DST";
gdalbuildvrt temp.vrt $SRC/*.tif;
# -co "TILED=YES" 
gdal_translate -ot Byte -of GTiff -co "TILED=YES" -co "BIGTIFF=IF_SAFER" -co "COMPRESS=LZW" temp.vrt "$DST";
rm temp.vrt
