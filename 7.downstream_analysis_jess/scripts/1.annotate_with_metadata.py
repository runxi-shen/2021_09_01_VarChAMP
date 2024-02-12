'''
Annotate and perform well-postion and cell count regression.
'''
import os
import pathlib
from typing import Optional

from concurrent import futures
import pandas as pd
import polars as pl
from tqdm import tqdm
import time

def annotate_with_platemap(profile_path: str, platemap_path: str, output_file_path: Optional[str] = None):
    """
    Annotate lazyframe using platemap

    Parameters
    ----------
    profile_path : str
        Dataframe file location
        
    platemap_path : str
        Platemap file location

    Returns
    -------
    Annotated polars LazyFrame of profiles
    """

    profile = pl.scan_parquet(profile_path)
    platemap = pl.scan_csv(platemap_path)
    
    # Append 'Metadata_' to platemap column names 
    # Create a mapping of old column names to new column names
    column_mapping = {col: f"Metadata_{col}" if "Metadata_" not in col else col for col in platemap.columns}

    # Apply the column mapping to the lazy frame
    platemap = platemap.select(
        *[pl.col(col).alias(new_col) for col, new_col in column_mapping.items()]
    )

    # Annotate with platemap
    merged_lf = profile.join(platemap, on = ["Metadata_Plate", "Metadata_Well"], how = "left")

    # Save or return dataframe
    if output_file_path != None:
        merged_lf.sink_parquet(path=output_file_path, 
                              compression="gzip")
        print(f'\n Annotated profile saved at {output_file_path}')
        
    else: 
        return merged_lf
    
def main():
    """Annotate and aggregate plate-level profiles.
    """
    
    batch_name = 'B1A1R1'

    # Input directories
    data_dir = pathlib.Path("/dgx1nas1/storage/data/jess/varchamp/sc_data/raw_profiles").resolve(strict=True)
    result_dir = pathlib.Path("/dgx1nas1/storage/data/jess/varchamp/sc_data/processed_profiles")
    result_dir.mkdir(exist_ok=True)

    # Input files
    platemap_file = f'/dgx1nas1/storage/data/jess/varchamp/platemaps/{batch_name}.csv'

    # Output file paths
    anot_file = pathlib.Path(result_dir / f'{batch_name}_annotated.parquet')
    
    # Annotate profiles
    all_paths = os.listdir(data_dir)
    plate_list = []
    for file in tqdm(all_paths):
        if not file.endswith(".parquet"): continue
        orig_file = pathlib.Path(data_dir / file).resolve(strict=True)
        df_ann = annotate_with_platemap(orig_file, platemap_file)
        plate_list.append(df_ann)

    # # Aggregate all profiles from batch and save
    lf_agg = pl.concat(plate_list)
    lf_agg.sink_parquet(path=anot_file, compression="gzip")
    
if __name__ == "__main__":
    main()