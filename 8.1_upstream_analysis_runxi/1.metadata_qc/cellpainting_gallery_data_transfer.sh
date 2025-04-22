#!/bin/bash

BASEPATH="s3://cellpainting-gallery/cpg0020-varchamp/broad/workspace"
HOMEPATH="/home/shenrunx/igvf/varchamp"

## Download data from cell painting gallery
## ====================================================================================================================== ##
## To download all of the sources in the JUMP dataset
## batches=`aws s3 ls --no-sign-request "$BASEPATH/backend/" | awk '{print substr($2, 1, length($2)-1)}'`

## download batch 7-8 data
# batches="2024_01_23_Batch_7 2024_02_06_Batch_8"

## download batch 13-14 data
# batches="2025_01_27_Batch_13 2025_01_28_Batch_14"
# mkdir -p inputs/metadata/platemaps

# for batch_id in $batches;
# do
    # aws s3 sync --no-sign-request "$BASEPATH/metadata/platemaps/$batch_id" inputs/platemaps/$batch_id
# done
## ====================================================================================================================== ##


#### Upload data to S3
## ====================================================================================================================== ##
# export AWS_PROFILE=cpg_staging
# export AWS_REGION=us-east-1

# # Run the AWS CLI command and capture the output
# output=$(aws s3control get-data-access \
#   --account-id 309624411020 \
#   --target "s3://staging-cellpainting-gallery/cpg0020-varchamp/*" \
#   --permission READWRITE \
#   --privilege Default \
#   --duration-seconds 43200 \
#   --region $AWS_REGION)

# # Check if the command was successful
# if [ $? -eq 0 ]; then
#   # Parse the output to extract credentials
#   AccessKeyId=$(echo "$output" | jq -r '.Credentials.AccessKeyId')
#   SecretAccessKey=$(echo "$output" | jq -r '.Credentials.SecretAccessKey')
#   SessionToken=$(echo "$output" | jq -r '.Credentials.SessionToken')

#   echo $AccessKeyId
#   echo $SecretAccessKey
#   echo $SessionToken

#   # Export the credentials as environment variables
#   export AWS_ACCESS_KEY_ID=$AccessKeyId
#   export AWS_SECRET_ACCESS_KEY=$SecretAccessKey
#   export AWS_SESSION_TOKEN=$SessionToken

#   echo "AWS credentials have been set successfully."
# else
#   echo "Failed to get data access information."
#   exit 1
# fi

batches="2025_01_27_Batch_13 2025_01_28_Batch_14"
for batch_id in $batches;
do
    UPLOADPATH="$BASEPATH/metadata/platemaps/$batch_id"
    aws s3 cp \
        outputs/corrected_platemaps/$batch_id/ \
        "$UPLOADPATH" \
        --recursive \
        --exclude "*" \
        --include "*.txt" \
        --include "*.csv" \
        --profile jump-cp-role # --dryrun
done
## ====================================================================================================================== ##