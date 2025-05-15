#!/bin/bash

cp inputs/snakemake_files/Snakefile_batch15 .
nohup snakemake \
    --snakefile Snakefile_batch15 \
    --cores all &> outputs/snakemake_logs/snakemake_batch15.log

cp inputs/snakemake_files/Snakefile_batch16 .
nohup snakemake \
    --snakefile Snakefile_batch16 \
    --cores all &> outputs/snakemake_logs/snakemake_batch16.log


## Run batch 12
# cp inputs/snakemake_files/Snakefile_batch12 .
# nohup snakemake \
#     --snakefile Snakefile_batch12 \
#     --cores 256 &> outputs/snakemake_logs/snakemake_batch12_new_classification.log

# ## Run batch 11
# cp inputs/snakemake_files/Snakefile_batch11 .
# nohup snakemake \
#     --snakefile Snakefile_batch11 \
#     --cores 256 &> outputs/snakemake_logs/snakemake_batch11_new_classification.log

# cp inputs/snakemake_files/Snakefile_batch12_widefield .
# nohup snakemake \
#     --snakefile Snakefile_batch12_widefield \
#     --cores 256 &> outputs/snakemake_logs/snakemake_batch12_widefield_new_classification.log

# ## Run batch 11
# cp inputs/snakemake_files/Snakefile_batch11_widefield .
# nohup snakemake \
#     --snakefile Snakefile_batch11_widefield \
#     --cores 256 &> outputs/snakemake_logs/snakemake_batch11_widefield_new_classification.log

# ## Run batch 8
# cp inputs/snakemake_files/Snakefile_batch8 .
# nohup snakemake \
#     --snakefile Snakefile_batch8 \
#     --cores 256 &> outputs/snakemake_logs/snakemake_batch8.log

# ## Run batch 7
# cp inputs/snakemake_files/Snakefile_batch7 .
# nohup snakemake \
#     --snakefile Snakefile_batch7 \
#     --cores 256 &> outputs/snakemake_logs/snakemake_batch7.log


# ## Run batch 13
# cp inputs/snakemake_files/Snakefile_batch13 .
# nohup snakemake \
#     --snakefile Snakefile_batch13 \
#     --cores all &> outputs/snakemake_logs/snakemake_batch13.log &

# ## Run batch 14
# cp inputs/snakemake_files/Snakefile_batch14 .
# nohup snakemake \
#     --snakefile Snakefile_batch14 \
#     --cores all &> outputs/snakemake_logs/snakemake_batch14.log &

