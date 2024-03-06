'''
Perform batch correction.
'''
import pathlib
import logging 

logging.basicConfig(format='%(levelname)s:%(asctime)s:%(name)s:%(message)s',
                    level=logging.INFO)

import pandas as pd 
from pycytominer.operations.transform import RobustMAD
import time
import polars as pl
import math

def filter_nans(df_to_filt: pl.DataFrame):
    
    feat_cols = [i for i in df_to_filt.columns if "Metadata_" not in i] 
    meta_cols = [i for i in df_to_filt.columns if "Metadata_" in i]

    # remove a row if >90% of features are NaNs
    num_rows = df_to_filt.shape[0]
    tol_num = len(meta_cols) + math.ceil(0.05*len(feat_cols))
    
    # import plotext as plt
    # nan_per_cell = df_to_filt.select(nans = pl.sum_horizontal(pl.all().is_null()))
    # plt.clear_figure()
    # plt.hist(nan_per_cell[["nans"]].to_series(), 10, label = "NaNs per cell")
    # plt.show()
    
    df_to_filt = df_to_filt.filter(pl.sum_horizontal(pl.all().is_null()) < tol_num)
    num_profiles_filtered = num_rows - df_to_filt.shape[0]

    # separate feature and meta
    df_meta = df_to_filt.select(meta_cols)
    df_to_filt = df_to_filt.drop(meta_cols)

    # keep a column (feature) only if there are 0 nulls
    num_cols = df_to_filt.shape[1]
    df_to_filt = df_to_filt[[s.name for s in df_to_filt if (s.null_count() == 0)]]
    num_feats_filtered = num_cols - df_to_filt.shape[1]

    # join back with df_meta
    df_to_filt = pl.concat([df_meta, df_to_filt], how = "horizontal")

    print(f'{num_profiles_filtered} profiles (rows) were filtered because they had greater than 5% NaNs. {num_feats_filtered} features (columns) were filtered because they contained at least 1 NaN')

    return df_to_filt
    

def main():
    print("Script started!")
    epsilon_mad = 0.0
    batch_name = 'B6A4R2'

    # Data directories
    data_dir = pathlib.Path(f"/dgx1nas1/storage/data/jess/varchamp/sc_data/processed_profiles/{batch_name}").resolve(strict=True)
    result_dir = data_dir

    # Input file paths
    anno_file = pathlib.Path(data_dir / f"{batch_name}_annotated.parquet")
    
    # Output file paths
    norm_file = pathlib.Path(result_dir / f"{batch_name}_annotated_corrected_normalized.parquet")
    
    # Filter NaNs
    df = pl.read_parquet(anno_file) 
    
    # 5678 features, 5390 filtered out if using 90% NaN filter. Why so many features? No cells had >90% NaNs so these are all "real" features
    
    # from histogram of NaNs per cell, more appropriate filter is 5% NaNs. 
    df = filter_nans(df) # this results in filtering out 150 cells and 251 features.
    
    # Well position correction
    feature_cols = [i for i in df.columns if "Metadata_" not in i] 
    df = df.with_columns(pl.col(feature_cols) - pl.mean(feature_cols).over("Metadata_Well"))
    df = filter_nans(df)

    # perform MAD normalization
    df = df.to_pandas()
    plate_list = df['Metadata_Plate'].unique().tolist()
    
    start = time.perf_counter()
    normalizer = RobustMAD(epsilon_mad)
    result_list = []
    for plate in plate_list:
        print(plate)

        df_plate = df[df['Metadata_Plate']==plate].copy()
        feat_cols = [i for i in df_plate.columns if "Metadata_" not in i] 
        meta_cols = [i for i in df_plate.columns if "Metadata_" in i] 

        meta = df_plate[meta_cols]
        normalizer.fit(df_plate[feat_cols])
        norm_feats = normalizer.transform(df_plate[feat_cols])
        norm_feats = pd.DataFrame(norm_feats,
                                    index=df_plate.index,
                                    columns=feat_cols)
        df_plate = pd.concat([meta, norm_feats], axis=1)
        result_list.append(df_plate)
    end = time.perf_counter()
    print(f'RobustMAD runtime: {end-start} secs.')
    
    df_norm = pd.concat(result_list, ignore_index=True)
    df_norm = filter_nans(pl.from_pandas(df_norm)) # 0 cells filtered out. 111 additional features filtered out. 
    
    df_norm.write_parquet(norm_file, compression="gzip")

if __name__ == '__main__':
    main()