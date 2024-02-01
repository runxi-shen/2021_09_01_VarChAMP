'''
Annotate and perform well-postion and cell count regression.
'''
import os
import pathlib
from typing import Optional

from concurrent import futures
import pandas as pd
from tqdm import tqdm
import time

from utils import get_features

def annotate_with_platemap(profile_path: str, platemap_path: str, output_file_path: Optional[str] = None):
    """
    Annotate dataframe using platemap

    Parameters
    ----------
    profile_path : str
        Dataframe file location
        
    platemap_path : str
        Platemap file location

    Returns
    -------
    Annotated pandas DataFrame of profiles
    """
    
    profile = pd.read_parquet(profile_path, engine="pyarrow")
    platemap = pd.read_csv(platemap_path).copy()
    
    # Append 'Metadata_' to platemap column names 
    platemap.columns = [
            f"Metadata_{x}" if not x.startswith("Metadata_") else x
            for x in platemap.columns
        ]

    # Annotate with platemap
    aligned_df = platemap.merge(profile, 
                                on=["Metadata_Plate", "Metadata_Well"], 
                                how="right")

    # Save or return dataframe
    if output_file_path != None:
        aligned_df.to_parquet(path=output_file_path, 
                              compression="gzip")
        print(f'\n Annotated profile saved at {output_file_path}')
        
    else: 
        return aligned_df
    
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
    plate_list = []
    for file in tqdm(os.listdir(data_dir)):
        if not file.endswith(".parquet"): continue
        orig_file = pathlib.Path(data_dir / file).resolve(strict=True)
        df_ann = annotate_with_platemap(orig_file, platemap_file)
        plate_list.append(df_ann)

    # # Aggregate all profiles from batch and save
    df_agg = pd.concat(plate_list, ignore_index=True)
    df_agg.to_parquet(path=anot_file, compression="gzip", index=False)
    
if __name__ == "__main__":
    main()