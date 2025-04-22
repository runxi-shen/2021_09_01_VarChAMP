#!/bin/bash

nohup snakemake \
    --snakefile Snakefile_batch13 \
    --cores all &> outputs/snakemake_logs/snakemake_batch13.log &
    
# nohup snakemake -s Snakefile_batch14 --cores all &> snakemake_batch14.log &