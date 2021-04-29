#!/usr/bin/env sh
# A bash script to parse dataset

# Parse inline input
DIR=$1
if [ -z "$DIR" ]; then
    echo "`date`: Usage: $0 <Directory to download>"
    exit 1
fi

echo "Download dataset to $DIR/north.zip"
curl -o "$DIR/north.zip" https://www.dropbox.com/s/avs19pzvwsvel0f/north.zip -L
