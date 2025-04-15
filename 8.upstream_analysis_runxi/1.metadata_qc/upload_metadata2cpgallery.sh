#!/bin/bash

export AWS_PROFILE=cpg_staging
export AWS_REGION=us-east-1

# Run the AWS CLI command and capture the output
output=$(aws s3control get-data-access \
  --account-id 309624411020 \
  --target "s3://staging-cellpainting-gallery/cpg0020-varchamp/*" \
  --permission READWRITE \
  --privilege Default \
  --duration-seconds 43200 \
  --region $AWS_REGION)

# Check if the command was successful
if [ $? -eq 0 ]; then
  # Parse the output to extract credentials
  AccessKeyId=$(echo "$output" | jq -r '.Credentials.AccessKeyId')
  SecretAccessKey=$(echo "$output" | jq -r '.Credentials.SecretAccessKey')
  SessionToken=$(echo "$output" | jq -r '.Credentials.SessionToken')

  echo $AccessKeyId
  echo $SecretAccessKey
  echo $SessionToken

  # Export the credentials as environment variables
  export AWS_ACCESS_KEY_ID=$AccessKeyId
  export AWS_SECRET_ACCESS_KEY=$SecretAccessKey
  export AWS_SESSION_TOKEN=$SessionToken

  echo "AWS credentials have been set successfully."
else
  echo "Failed to get data access information."
  exit 1
fi


## Data paths
BASEPATH="s3://staging-cellpainting-gallery/cpg0020-varchamp/broad/workspace"
# HOMEPATH="/home/shenrunx/igvf/varchamp"

# ## Download data from S3
# batch_id="2024_01_23_Batch_7"
# aws s3 sync --no-sign-request "$BASEPATH/metadata/$batch_id" ../input/meta_correct_batch78/$batch_id

# batch_id="2024_02_06_Batch_8"
# aws s3 sync --no-sign-request "$BASEPATH/metadata/$batch_id" ../input/meta_correct_batch78/$batch_id

## Upload data to S3
batches="2024_01_23_Batch_7 2024_02_06_Batch_8"
for batch_id in $batches;
do
    UPLOADPATH="$BASEPATH/metadata/platemaps/$batch_id/platemap/"
    aws s3 cp ../../output/meta_correct_batch78/$batch_id/ "$UPLOADPATH" --recursive --exclude "*" --include "*.txt"
done

## Example to update Cell Painting gallery directly
aws s3 cp --profile jump-cp-role x s3://cellpainting-gallery/cpg0020-varchamp/broad/workspace/temp/x