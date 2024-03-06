'''
Annotate and perform well-postion and cell count regression.
'''
import os
import pathlib
from typing import Optional
import polars as pl
from tqdm import tqdm

def annotate_with_platemap(profile_path: str, platemap_path: str, platemap_sep: str, output_file_path: Optional[str] = None):
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
    platemap = pl.scan_csv(platemap_path, separator = platemap_sep)
    
    # Append 'Metadata_' to platemap column names 
    # Create a mapping of old column names to new column names
    column_mapping = {col: f"Metadata_{col}" if "Metadata_" not in col else col for col in platemap.columns}

    # Apply the column mapping to the lazy frame
    platemap = platemap.select(
        *[pl.col(col).alias(new_col) for col, new_col in column_mapping.items()]
    )
    platemap = platemap.rename({"Metadata_well_position": "Metadata_Well"})

    # Annotate with platemap
    merged_lf = profile.join(platemap, on = ["Metadata_Well"], how = "left")

    # Save or return dataframe
    if output_file_path != None:
        merged_df = merged_lf.collect()
        merged_df.write_parquet(output_file_path, compression="gzip")
        print(f'\n Annotated profile saved at {output_file_path}')
        
    else: 
        return merged_lf
    
def main():
    """Annotate and aggregate plate-level profiles.
    """
    
    batch_name = 'B6A4R2'

    # Input directories
    data_dir = pathlib.Path(f"/dgx1nas1/storage/data/jess/varchamp/sc_data/raw_profiles/{batch_name}").resolve(strict=True)
    result_dir = pathlib.Path(f"/dgx1nas1/storage/data/jess/varchamp/sc_data/processed_profiles/{batch_name}")
    result_dir.mkdir(exist_ok=True)

    # Input files
    platemap_file = f'/dgx1nas1/storage/data/jess/varchamp/platemaps/{batch_name}.txt'

    # Output file paths
    anot_file = pathlib.Path(result_dir / f'{batch_name}_annotated.parquet')
    
    # Annotate profiles
    all_paths = os.listdir(data_dir)
    plate_list = []
    for file in tqdm(all_paths):
        if not file.endswith(".parquet"): continue
        orig_file = pathlib.Path(data_dir / file).resolve(strict=True)
        df_ann = annotate_with_platemap(orig_file, platemap_file, "\t")
        plate_list.append(df_ann)

    # Aggregate all profiles from batch
    lf_agg = pl.concat(plate_list)

    # Re-order columns so all metadata are on the left
    metadata_columns = [col for col in lf_agg.columns if "Metadata_" in col]
    other_columns = [col for col in lf_agg.columns if "Metadata_" not in col]
    reordered_columns = metadata_columns + other_columns
    lf_agg = lf_agg.select(*reordered_columns)

    # Convert to dataframe and write out to parquet
    df_agg = lf_agg.collect()
    print('finished aggregating')
    df_agg.write_parquet(anot_file, compression="gzip")
    
if __name__ == "__main__":
    main()