#!/usr/bin/env sh
# A bash script to parse dataset

# Parse inline input
DIR=$1
if [ -z "$DIR" ]; then
    echo "`date`: Usage: $0 <Directory to download> <Type to download. \
    {1: base dataset: validate and other necessary catalogs, \
    2: perfect train, 3: train with quality score 9, \
    4: train with quality score 8}>"
    exit 1
fi

TYPE=$2
if [ -z "$TYPE" ]; then
    echo "`date`: Usage: $0 <Directory to download>"
    exit 1
fi

echo "Download dataset to $DIR/dl_data.zip"
curl -o "$DIR/north.zip" https://www.dropbox.com/s/bz3zcomjp6hkshw/dl_data.zip -L
unzip $DIR/dl_data.zip -d $DIR
rm -rf $DIR/dl_data.zip
rm -rf $DIR/__MACOSX

