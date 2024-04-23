"""Annotate dataframe with platemap"""
import pandas as pd
import sys
sys.path.append('..')
from utils import remove_nan_infs_columns

def get_platemap(csv_path: str, plate: str) -> str:
    """Get the platemap .txt file"""
    barcode = pd.read_csv(csv_path)
    barcode_dict = dict(zip(barcode["Assay_Plate_Barcode"], barcode["Plate_Map_Name"]))
    return barcode_dict[plate]

def annotate_with_platemap(
    profile_path: str, platemap_path: str, output_file_path: str
):
    """Annotate dataframe using platemap"""
    
    profile = pd.read_parquet(profile_path, engine="pyarrow")
    platemap = pd.read_csv(platemap_path, sep="\t").copy()

    if "Metadata_Well" not in platemap.columns:
        platemap["Metadata_Well"] = platemap["well_position"]

    # Append 'Metadata_' to platemap column names
    platemap.columns = [
        f"Metadata_{x}" if not x.startswith("Metadata_") else x
        for x in platemap.columns
    ]

    # Annotate with platemap
    aligned_df = platemap.merge(profile, on=["Metadata_Well"], how="right")

    aligned_df.to_parquet(path=output_file_path, compression="gzip", index=False)

def aggregate(plate_list: list[str], output_path: str):
    """Aggregate profiles and save"""
    df_plate_list = []
    for file in plate_list:
        if not file.endswith(".parquet"):
            continue
        df = pd.read_parquet(file)
        df_plate_list.append(df)
    df_agg = pd.concat(df_plate_list, ignore_index=True)
    df_agg.to_parquet(path=output_path, compression="gzip", index=False)

def filter_nan(input_path: str, output_path: str):
    '''Filter nan and inf columns'''
    dframe = pd.read_parquet(input_path)
    dframe = remove_nan_infs_columns(dframe)
    dframe.to_parquet(output_path, compression="gzip", index=False)
    