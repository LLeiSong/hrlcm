#!/usr/bin/env sh
# Authors: Lyndon Estes, Boka Luo, Lei Song
## create a spot instance from AMI
## Customize for a macOS
## Before run this script, do:
## 1. brew install awscli, then aws configure to configure aws credential.
## 2. brew install jq
## 3. brew install coreutils. to use gdata

# Get input parameters
AMIID=$1
if [ -z "$AMIID" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <new_instance_name> <spot_type> <valid_until> <key_name> <volume_size> \
    <zone>"
    exit 1
fi
ITYPE=$2
if [ -z "$ITYPE" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <new_instance_name> <spot_type> <valid_until> <key_name> <volume_size> \
    <zone>"
    exit 1
fi
SECGROUPID=$3
if [ -z "$SECGROUPID" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <new_instance_name> <spot_type> <valid_until> <key_name> <volume_size> \
    <zone>"
    exit 1
fi
NEWINAME=$4
if [ -z "$NEWINAME" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <new_instance_name> <spot_type> <valid_until> <key_name> <volume_size> \
    <zone>"
    exit 1
fi
SPOTTYPE=$5
if [ -z "$SPOTTYPE" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <new_instance_name> <spot_type> <valid_until> <key_name> <volume_size> \
    <zone>"
    exit 1
fi
VALIDUNTIL=$6
if [ -z "$VALIDUNTIL" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <new_instance_name> <spot_type> <valid_until> <key_name> <volume_size> \
    <zone>"
    exit 1
fi
KEYNAME=$7
if [ -z "$KEYNAME" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <new_instance_name> <spot_type> <valid_until> <key_name> <volume_size> \
    <zone>"
    exit 1
fi
SDASIZE=$8
if [ -z "$SDASIZE" ]; then
    echo "`date`: Usage: $0 <ami_id> <instance_type> <security_group_id> \
    <new_instance_name> <spot_type> <valid_until> <key_name> <volume_size> \
    <zone>"
    exit 1
fi
ZONE=$9

# get bid price
START_TIME=$(gdate --date="3 days ago" +"%Y-%m-%dT%T")
END_TIME=$(gdate +"%Y-%m-%dT%T")

read -r -d '' PRICES << EOF
    $(aws ec2 describe-spot-price-history --instance-types $ITYPE \
		--product-description Linux/UNIX \
		--start-time $START_TIME \
		--end-time $END_TIME)
EOF

## find lowest price zone and get the max spot price in that zone
if [ -z "$ZONE" ]; then
  echo "Find lowest price zone"
  ZONE=$(echo $PRICES |\
    jq '.SpotPriceHistory| sort_by(.AvailabilityZone | explode | map(-.)) |
    min_by(.SpotPrice | tonumber)|.AvailabilityZone')
else
  ZONE=$(echo "\"$ZONE\"")
fi
echo "$ZONE"

MAX_SPOT_PRICE=$(echo $PRICES |\
	jq '[.SpotPriceHistory[] | select(.AvailabilityZone == '"$ZONE"')] | max_by(.SpotPrice | tonumber) |.SpotPrice |tonumber')


## get bid price by adding an overflow
OVERFLOW=0.02
BID_PRICE=$(echo | awk -v a=$MAX_SPOT_PRICE -v b=$OVERFLOW '{print a+b}')

## get subnetId of lowest price zone
SUBNETID=$(aws ec2 describe-subnets \
		--filter 'Name=availability-zone,Values='$ZONE'' \
		           'Name=vpc-id,Values=vpc-e48b1a9d' \
		--output text \
		--query 'Subnets[*].SubnetId')


## Set up new instance
echo "Setting up new spot instance named $NEWINAME from AMI $AMIID with volume size $SDASIZE in $ZONE on a bid_price of $BID_PRICE"

if [ "$SPOTTYPE" == "persistent" ]; then
  aws ec2 run-instances \
    --image-id $AMIID \
    --count 1 \
    --instance-type $ITYPE \
    --subnet-id $SUBNETID \
    --iam-instance-profile 'Name="activemapper_planet_readwriteS3"' \
    --key-name $KEYNAME \
    --security-group-ids $SECGROUPID \
    --block-device-mappings \
    "[ { \"DeviceName\": \"/dev/sda1\", \"Ebs\": { \"VolumeSize\": $SDASIZE } } ]" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value='$NEWINAME'}]' \
    --instance-market-options 'MarketType=spot,
          SpotOptions={MaxPrice='$BID_PRICE',
          SpotInstanceType='$SPOTTYPE',
          ValidUntil='$VALIDUNTIL',
          InstanceInterruptionBehavior=stop}'
else
  aws ec2 run-instances \
    --image-id $AMIID \
    --count 1 \
    --instance-type $ITYPE \
    --subnet-id $SUBNETID \
    --iam-instance-profile 'Name="activemapper_planet_readwriteS3"' \
    --key-name $KEYNAME \
    --security-group-ids $SECGROUPID \
    --block-device-mappings \
    "[ { \"DeviceName\": \"/dev/sda1\", \"Ebs\": { \"VolumeSize\": $SDASIZE } } ]" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value='$NEWINAME'}]' \
    --instance-market-options 'MarketType=spot,
          SpotOptions={MaxPrice='$BID_PRICE',
          SpotInstanceType='$SPOTTYPE',
          InstanceInterruptionBehavior=terminate}'
fi

NEWIID=$(aws ec2 describe-instances \
	--filters 'Name=tag:Name,Values='"$NEWINAME"'' \
	--output text --query 'Reservations[*].Instances[*].InstanceId')


echo $NEWIID

# ./tools/create_spot_instance.sh ami-0833ca42c91eb4a58 g4dn.12xlarge \
# sg-0a8bbc91697d6a76b tzlcms persistent 2021-05-15T23:00:00 lsong-keypair 200