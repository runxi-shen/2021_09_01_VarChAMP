import pathlib
import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
# ignore mix type warnings from pandas
import warnings
warnings.filterwarnings("ignore")

from pycytominer import feature_select


   
# Setting file paths
data_dir = pathlib.Path("/dgx1nas1/storage/data/sam/processed").resolve(strict=True)
output_path = pathlib.Path(data_dir / '2023_05_30_B1A1R1_normalized_feature_selected.parquet')


# List all plates
plate_list = [i for i in os.listdir(data_dir) if 'normalized' in i]

df_list = []
# Concatenate all profiles and perform feature selection
for file in tqdm(plate_list):
    norm_file = pathlib.Path(data_dir / file).resolve(strict=True)
    df_list.append(pd.read_parquet(norm_file))

df_all_profiles = pd.concat(df_list, ignore_index=True)

print('Feature selection started.')
# Perform batch level feature selection 
feature_select(
                profiles=df_all_profiles,
                features="infer",
                image_features=False,
                samples="all",
                operation=["variance_threshold", "correlation_threshold", "blocklist", "drop_na_columns"],
                output_file=output_path,
                output_type='parquet',
                compression_options="gzip",
            )

print(f"Feature selected profile saved in: {output_path}")
    