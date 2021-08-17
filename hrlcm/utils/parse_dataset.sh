#!/usr/bin/env sh
# A bash script to parse dataset

# Parse inline input
DIR=$1
if [ -z "$DIR" ]; then
    echo "`date`: Usage: $0 <Directory to download>"
    exit 1
fi

echo "Download dataset to $DIR/dl_data.zip"
curl -o "$DIR/dl_data.zip" https://www.dropbox.com/s/ghlssjk7ojoj6ob/dl_train_valid.zip -L
unzip $DIR/dl_data.zip -d $DIR
rm -rf $DIR/dl_data.zip
rm -rf $DIR/__MACOSX

