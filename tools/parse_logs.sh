#!/usr/bin/env sh
# Authors: Lei Song
## Parse log files (arguments, checkpoints, and scalar)

# Get input parameters
DNS=$1
if [ -z "$DNS" ]; then
    echo "`date`: Usage: $0 <Public_IPv4_DNS> <experiment_name> <download_dir>"
    exit 1
fi
ENAME=$2
if [ -z "$ENAME" ]; then
    echo "`date`: Usage: $0 <Public_IPv4_DNS> <experiment_name> <download_dir>"
    exit 1
fi
DIR=$3
if [ -z "$DIR" ]; then
    echo "`date`: Usage: $0 <Public_IPv4_DNS> <experiment_name> <download_dir>"
    exit 1
fi

# Download files in loop
while true; do scp -r ubuntu@$DNS:~/hrlcm/results/dl/$ENAME $DIR; sleep 2000; done
