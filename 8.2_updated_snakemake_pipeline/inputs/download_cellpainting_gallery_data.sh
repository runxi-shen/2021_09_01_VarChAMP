#!/bin/bash

BASEPATH="s3://cellpainting-gallery/cpg0020-varchamp/broad/workspace"

# HOMEPATH="/home/shenrunx/igvf/varchamp"
## To download all of the sources in the JUMP dataset
## batches=`aws s3 ls --no-sign-request "$BASEPATH/backend/" | awk '{print substr($2, 1, length($2)-1)}'`

## download batch 7-8 data
# batches="2024_01_23_Batch_7 2024_02_06_Batch_8"

## download batch 11-12 data
batches="2024_12_09_Batch_11 2024_12_09_Batch_12"

## download batch 13-14 data
# batches="2025_01_27_Batch_13 2025_01_28_Batch_14"

## download batch 15-16 data
# batches="2025_03_17_Batch_15,2025_03_17_Batch_16"

# mkdir -p inputs/metadata/platemaps

for batch_id in $batches;
do
    aws s3 sync --no-sign-request --exclude "*.csv" "$BASEPATH/backend/$batch_id" single_cell_profiles/$batch_id
    # aws s3 sync --no-sign-request "$BASEPATH/metadata/platemaps/$batch_id" inputs/metadata/platemaps/$batch_id

    ## creating symbolic links if needed
    # ln -s $HOMEPATH/2021_09_01_VarChAMP-data/profiles/$batch_id inputs/well_profiles/$batch_id
    # ln -s $HOMEPATH/2021_09_01_VarChAMP-data/metadata/platemaps/$batch_id inputs/metadata/platemaps/$batch_id

    ## For some future reference when we want to switch from sqlite to .csv's directly
    # aws s3 cp --no-sign-request \
    #     $BASEPATH/analysis/$batch_id/2025_01_27_B13A7A8P1_T1/analysis/ \
    #     inputs/analysis/2025_01_27_B13A7A8P1_T1/ \
    #     --exclude "*" \
    #     --include "**/Cells.csv" \
    #     --include "**/Cytoplasm.csv" \
    #     --include "**/Nuclei.csv" \
    #     --include "**/Image.csv" \
    #     --recursive 
done
