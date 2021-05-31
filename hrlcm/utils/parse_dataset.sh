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

if [ "$TYPE" == "1" ]; then
  echo "Download dataset to $DIR/north.zip"
  curl -o "$DIR/north.zip" https://www.dropbox.com/s/v005zhcxwxfnf16/north.zip -L
elif [ "$TYPE" == "2" ]; then
  echo "Download dataset to $DIR/dl_train.zip"
  curl -o "$DIR/dl_train.zip" https://www.dropbox.com/s/gz463kp9yxtuqef/dl_train.zip -L
elif [ "$TYPE" == "3" ]; then
  echo "Download dataset to $DIR/dl_train_score9.zip"
  curl -o "$DIR/dl_train_score9.zip" https://www.dropbox.com/s/j4lozxcvxe4yvs6/dl_train_score9.zip -L
elif [ "$TYPE" == "4" ]; then
  echo "Download dataset to $DIR/dl_train_score8.zip"
  curl -o "$DIR/dl_train_score8.zip" https://www.dropbox.com/s/j4lozxcvxe4yvs6/dl_train_score8.zip -L
else
  echo "No such type."
fi
