#!/bin/bash

## Run batch 13
cp inputs/snakemake_files/Snakefile_batch13 .
nohup snakemake \
    --snakefile Snakefile_batch13 \
    --cores all &> outputs/snakemake_logs/snakemake_batch13.log &

## Run batch 14
cp inputs/snakemake_files/Snakefile_batch14 .
nohup snakemake \
    --snakefile Snakefile_batch14 \
    --cores all &> outputs/snakemake_logs/snakemake_batch14.log &