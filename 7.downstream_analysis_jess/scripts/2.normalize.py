'''
Perform batch correction.
'''
import pathlib
import logging 

logging.basicConfig(format='%(levelname)s:%(asctime)s:%(name)s:%(message)s',
                    level=logging.INFO)

from concurrent import futures
from typing import Optional
import pandas as pd 
from tqdm import tqdm
from statsmodels.formula.api import ols
from sklearn.base import TransformerMixin
from pycytominer.operations.transform import RobustMAD
import time
from copairs.map import average_precision
import numpy as np
import polars as pl
from datetime import datetime
import os

from utils import get_features


def regress_out_cell_counts(
    well_df: pd.DataFrame,
    sc_df: pd.DataFrame,
    cc_col: str,
    min_unique: int = 100,
    cc_rename: Optional[str] = None,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Regress out cell counts from all features in a dataframe.

    Parameters
    ----------
    well_df : pandas.core.frame.DataFrame
        DataFrame of well-level profiles.
    sc_df : pandas.core.frame.DataFrame
        DataFrame of single-cell profiles.
    cc_col : str
        Name of column containing cell counts.
    min_unique : int, optional
        Minimum number of unique feature values to perform regression.
    cc_rename : str, optional
        Name to rename cell count column to.
    inplace : bool, optional
        Whether to perform operation in place.

    Returns
    -------
    df : pandas.core.frame.DataFrame
    """
    df = well_df if inplace else well_df.copy()
    df_sc = sc_df if inplace else sc_df.copy()
 
    feature_cols = list(set(get_features(df)) & set(get_features(df_sc)))
    feature_cols = [
        feature for feature in feature_cols if df[feature].nunique() > min_unique
    ]

    def regress_out_cell_counts_parallel_helper(feature):
        model = ols(f"{feature} ~ {cc_col}", data=df).fit()
        return {feature: model.resid}

    with futures.ThreadPoolExecutor() as executor:
        results = executor.map(regress_out_cell_counts_parallel_helper, feature_cols)
    
    for res in results:
        [[feature, _]] = res.items() 
        new_df = df[['Metadata_Plate', 'Metadata_Well']]
        new_df = pd.concat([new_df, pd.DataFrame(res)])
        df_merged = df_sc[['Metadata_Plate', 'Metadata_Well', feature]].merge(new_df, 
                                                                              on=['Metadata_Plate', 'Metadata_Well'], 
                                                                              suffixes=("", "_mean"))
        df_sc[feature] = df_sc[feature] - df_merged[f'{feature}_mean']

    return df_sc

def prep_for_map(df_path: str, map_cols: [str], sample_col: [str], sample_n: int = 5): # type: ignore

    # define filters
    q = pl.scan_parquet(df_path).filter(
        (pl.col("Metadata_node_type") != "TC") &  # remove transfection controls
        (pl.col("Metadata_node_type") != "NC") &
        (pl.col("Metadata_node_type") != "PC") &
        (pl.col("Metadata_node_type") != "CC") &
        (pl.col("Metadata_allele") != "_NA") & 
        (pl.sum_horizontal(pl.col(map_cols).is_null()) == 0)  # remove any row with missing values for selected meta columns
        ).with_columns(pl.concat_str(sample_col).alias('Metadata_samplecol'))
    
    # if a sample column name was provided, randomly sample sample_n rows from each column category
    if sample_col:
        q = q.filter(pl.int_range(0, pl.len()).shuffle().over('Metadata_samplecol') < sample_n)
    
    # different data frames for metadata and profiling data
    map_cols_id = map_cols.copy()
    map_cols_id.append("Metadata_CellID")
    meta_cols = q.select(map_cols_id)
    meta_df = meta_cols.collect().to_pandas()

    feat_col = [i for i in q.columns if "Metadata_" not in i] 
    q = q.select(feat_col)
    feat_df = q.collect().to_pandas()

    map_input = {'meta': meta_df, 'feats': feat_df}

    return map_input
    


def main():
    print("Script started!")
    epsilon_mad = 0.0
    batch_name = 'B1A1R1'

    # Data directories
    data_dir = pathlib.Path("/dgx1nas1/storage/data/jess/varchamp/sc_data/processed_profiles").resolve(strict=True)
    result_dir = data_dir

    # Input file paths
    anno_file = pathlib.Path(data_dir / f"{batch_name}_annotated_sam.parquet")
    anno_cellID = pathlib.Path(data_dir / f"{batch_name}_annotated_cellID.parquet")
    df_well_path = f'/dgx1nas1/storage/data/jess/varchamp/well_data/{batch_name}_well_level.parquet'

    # add a column with unique cell ID
    if not os.path.exists(anno_cellID):
        print("Creating cellID column!")
        lf = pl.scan_parquet(anno_file).with_columns(pl.concat_str([pl.col("Metadata_Plate"),
                                                                    pl.col("Metadata_Well"),
                                                                    pl.col("Metadata_ImageNumber"),
                                                                    pl.col("Metadata_ObjectNumber")],
                                                                    separator="_").alias("Metadata_CellID"))
        df = lf.collect()
        df.write_parquet(anno_cellID, compression="gzip")

    # Output file paths
    map_dir = pathlib.Path("/dgx1nas1/storage/data/jess/varchamp/sc_data/map_results/").resolve(strict=True)
    well_file = pathlib.Path(result_dir / f"{batch_name}_annotated_corrected_wellmean.parquet")
    cc_file = pathlib.Path(result_dir / f"{batch_name}_annotated_corrected_cc.parquet")
    norm_file = pathlib.Path(result_dir / f"{batch_name}_annotated_corrected_normalized.parquet")

    # Set paramters for mAP
    pos_sameby = ['Metadata_allele']
    pos_diffby = ['Metadata_Plate']
    neg_sameby = ['Metadata_Plate']
    neg_diffby = ['Metadata_allele']
    batch_size = 20000
    sample_n_cells = 5
    sample_neg = True

    # Initialize some variables
    df_well_corrected = None

    map_cols = list(set(pos_sameby + pos_diffby + neg_sameby + neg_diffby))

    # compute baseline map
    if not os.path.exists(f'{map_dir}/baseline_map.parquet'):
        start = time.perf_counter()
        print("start map prep")
        map_input = prep_for_map(anno_cellID, map_cols, ['Metadata_Well', 'Metadata_Plate'], sample_n_cells)
        print("start map")
        map_result = average_precision(map_input['meta'], map_input['feats'].values, pos_sameby, pos_diffby, neg_sameby, neg_diffby, 
                                    batch_size, sample_neg = sample_neg)
        map_result.to_parquet(path=f'{map_dir}/baseline_map.parquet', compression="gzip", index=False)
        end = time.perf_counter()
        print(f'baseline map runtime: {end-start}.')

    # Well position correction
    if not os.path.exists(well_file):
        print("starting well correction!")
        start = time.perf_counter()
        lf = pl.scan_parquet(anno_cellID)
        feature_cols = [i for i in lf.columns if "Metadata_" not in i] 
        lf = lf.with_columns(pl.col(feature_cols) - pl.mean(feature_cols).over("Metadata_Well"))
        df_well_corrected = lf.collect()
        df_well_corrected.write_parquet(well_file, compression="gzip")
        end = time.perf_counter()
        print(f'Well position correction runtime: {end-start}.')

    # Compute map after well position correction
    if not os.path.exists(f'{map_dir}/well_corrected_map.parquet'):
        print("starting map after well position correction!")
        start = time.perf_counter()
        map_input = prep_for_map(well_file, map_cols, ['Metadata_Well', 'Metadata_Plate'], sample_n_cells)
        map_result = average_precision(map_input['meta'], map_input['feats'].values, pos_sameby, pos_diffby, neg_sameby, neg_diffby, 
                                    batch_size, sample_neg = sample_neg)
        map_result.to_parquet(path=f'{map_dir}/well_corrected_map.parquet', compression="gzip", index=False)
        end = time.perf_counter()
        print(f'well position map runtime: {end-start}.')

    # perform MAD normalization
    if not os.path.exists(norm_file):
        print("Starting MAD normalization!")
        if df_well_corrected is None:
            df_well_corrected = pd.read_parquet(well_file)
        
        df_well_level = pd.read_parquet(df_well_path)
        plate_list = df_well_level['Metadata_Plate'].unique().tolist()
        
        normalizer = RobustMAD(epsilon_mad)
        result_list = []
        for plate in plate_list:
            print(plate)
    
            df_plate = df_well_corrected[df_well_corrected['Metadata_Plate']==plate].copy()
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

        start = time.perf_counter()
        end = time.perf_counter()
        print(f'RobustMAD runtime: {end-start} secs.')

        df_norm = pd.concat(result_list, ignore_index=True)
        df_norm.to_parquet(path=norm_file, compression="gzip", index=False)

    # compute map after MAD
    if not os.path.exists(f'{map_dir}/mad_normalized_map.parquet'):
        print("Starting MAP after MAD!")
        start = time.perf_counter()
        map_input = prep_for_map(norm_file, map_cols, ['Metadata_Well', 'Metadata_Plate'], sample_n_cells)
        map_result = average_precision(map_input['meta'], map_input['feats'].values, pos_sameby, pos_diffby, neg_sameby, neg_diffby, 
                                    batch_size, sample_neg = sample_neg)
        map_result.to_parquet(path=f'{map_dir}/mad_normalized_map.parquet', compression="gzip", index=False)
        end = time.perf_counter()
        print(f'robustMAD map runtime: {end-start}.')
    
    # Cell count regression
    if not os.path.exists(cc_file):
        print("Starting cell count correction!")
        start = time.perf_counter()
        df_well_level = pd.read_parquet(df_well_path)
        plate_list = df_well_level['Metadata_Plate'].unique().tolist()
        df_cc_corrected = regress_out_cell_counts(df_well_level, pd.read_parquet(norm_file),'Metadata_Object_Count')
        end = time.perf_counter()
        print(f'Cell count regression runtime: {end-start}.')
        df_cc_corrected.to_parquet(path=cc_file, compression="gzip", index=False)

    # compute map after cc regression
    if not os.path.exists(f'{map_dir}/cc_regression_map.parquet'):
        print("Starting MAP after CC correction!")
        start = time.perf_counter()
        map_input = prep_for_map(cc_file, map_cols, ['Metadata_Well', 'Metadata_Plate'], sample_n_cells)
        map_result = average_precision(map_input['meta'], map_input['feats'].values, pos_sameby, pos_diffby, neg_sameby, neg_diffby, 
                                    batch_size, sample_neg = sample_neg)
        map_result.to_parquet(path=f'{map_dir}/cc_regression_map.parquet', compression="gzip", index=False)
        end = time.perf_counter()
        print(f'cc regression map runtime: {end-start}.')

if __name__ == '__main__':
    main()