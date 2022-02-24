#!/usr/bin/env sh
# A bash script to parse dataset

# Parse inline input
DIR=$1
if [ -z "$DIR" ]; then
    echo "`date`: Usage: $0 <Directory to download>"
    exit 1
fi

echo "Download dataset to $DIR/tanzania_train_valid.zip"
curl -o "$DIR/tanzania_train_valid.zip" https://www.dropbox.com/s/1xvtyyx2a6jdovi/tanzania_train_valid.zip -L
unzip $DIR/tanzania_train_valid.zip -d $DIR
rm -rf $DIR/tanzania_train_valid.zip
rm -rf $DIR/__MACOSX

