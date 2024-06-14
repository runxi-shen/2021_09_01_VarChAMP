#!/bin/bash
BASEPATH="s3://cellpainting-gallery/cpg0020-varchamp/broad/workspace"
HOMEPATH="/dgx1nas1/storage/data/sam/codes"
# To download all of the sources in the JUMP dataset
# batches=`aws s3 ls --no-sign-request "$BASEPATH/backend/" | awk '{print substr($2, 1, length($2)-1)}'`

# batches="2023-12-15_B4A3R1 2023-12-15_B4A4R1 2023-12-21_B6A3R2 2023-12-22_B6A4R2"
batches="2024_01_23_Batch_7 2024_02_06_Batch_8"

mkdir -p inputs/well_profiles
mkdir -p inputs/metadata/platemaps

for batch_id in $batches;
do
    aws s3 sync --no-sign-request --exclude "*.csv" "$BASEPATH/backend/$batch_id" inputs/single_cell_profiles/$batch_id
    ln -s $HOMEPATH/2021_09_01_VarChAMP-data/profiles/$batch_id inputs/well_profiles/$batch_id
    ln -s $HOMEPATH/2021_09_01_VarChAMP-data/metadata/platemaps/$batch_id inputs/metadata/platemaps/$batch_id
done
