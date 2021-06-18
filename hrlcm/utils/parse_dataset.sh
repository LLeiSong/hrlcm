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
    echo "`date`: Usage: $0 <Directory to download> <Type to download. \
    {1: base dataset: validate and other necessary catalogs, \
    2: perfect train, 3: train with quality score 9, \
    4: train with quality score 8}>"
    exit 1
fi

if [ "$TYPE" = "1" ]; then
  echo "Download dataset to $DIR/north.zip"
  curl -o "$DIR/north.zip" https://www.dropbox.com/s/7cxlclajdo6wid0/north.zip -L
  unzip $DIR/north.zip -d $DIR
  rm -rf $DIR/north.zip
  rm -rf $DIR/__MACOSX
elif [ "$TYPE" = "2" ]; then
  echo "Download dataset to $DIR/dl_train.zip"
  curl -o "$DIR/dl_train.zip" https://www.dropbox.com/s/gz463kp9yxtuqef/dl_train.zip -L
  unzip $DIR/dl_train.zip -d $DIR
  rm -rf $DIR/dl_train.zip
  rm -rf $DIR/__MACOSX
elif [ "$TYPE" = "3" ]; then
  echo "Download dataset to $DIR/dl_train_score9.zip"
  curl -o "$DIR/dl_train_score9.zip" https://www.dropbox.com/s/j4lozxcvxe4yvs6/dl_train_score9.zip -L
  unzip $DIR/dl_train_score9.zip -d $DIR
  rm -rf $DIR/dl_train_score9.zip
  rm -rf $DIR/__MACOSX
elif [ "$TYPE" = "4" ]; then
  echo "Download dataset to $DIR/dl_train_score8.zip"
  curl -o "$DIR/dl_train_score8.zip" https://www.dropbox.com/s/pzc11fsh56lj0he/dl_train_score8.zip -L
  unzip $DIR/dl_train_score8.zip -d $DIR
  rm -rf $DIR/dl_train_score8.zip
  rm -rf $DIR/__MACOSX
else
  echo "No such type."
fi
