#!/bin/bash

BASEPATH="s3://cellpainting-gallery/cpg0020-varchamp/broad/workspace"

#### Download data from cell painting gallery
# ====================================================================================================================== ##
# To download all of the sources in the JUMP dataset
# batches=`aws s3 ls --no-sign-request "$BASEPATH/backend/" | awk '{print substr($2, 1, length($2)-1)}'`

## store the platemaps in relative paths
mkdir -p ../inputs/platemaps

## download batch 7-8 data
# batches="2024_01_23_Batch_7 2024_02_06_Batch_8"
# for batch_id in $batches;
# do
#     aws s3 cp --no-sign-request --recursive --exclude "*" --include "*.csv" --include "platemap/B*A1*.txt" "$BASEPATH/metadata/platemaps/$batch_id" ../inputs/platemaps/$batch_id
# done

## download batch 11-12 data
batches="2024_12_09_Batch_11 2024_12_09_Batch_12"

## download batch 13-14 data
# batches="2025_01_27_Batch_13 2025_01_28_Batch_14"

for batch_id in $batches;
do
    aws s3 sync --no-sign-request --exclude "*" --include "*.csv" --include "platemap/*.txt" "$BASEPATH/metadata/platemaps/$batch_id" ../inputs/platemaps/$batch_id
done
# ====================================================================================================================== ##


#### Upload data to S3
## ====================================================================================================================== ##
# batches="2025_01_27_Batch_13 2025_01_28_Batch_14"
# for batch_id in $batches;
# do
#     UPLOADPATH="$BASEPATH/metadata/platemaps/$batch_id"
#     aws s3 cp \
#         ../outputs/corrected_platemaps/$batch_id/ \
#         "$UPLOADPATH" \
#         --recursive \
#         --exclude "*" \
#         --include "*.txt" \
#         --include "*.csv" \
#         --profile jump-cp-role # --dryrun
# done

# batches="2025_03_17_Batch_15 2025_03_17_Batch_16"
# for batch_id in $batches;
# do
#     UPLOADPATH="$BASEPATH/metadata/platemaps/$batch_id"
#     aws s3 cp \
#         ../inputs/platemaps/$batch_id/ \
#         "$UPLOADPATH" \
#         --recursive \
#         --exclude "*" \
#         --include "*.txt" \
#         --include "*.csv" \
#         --profile jump-cp-role # --dryrun
# done
## ====================================================================================================================== ##