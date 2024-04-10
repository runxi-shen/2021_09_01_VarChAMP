"""Perform correction before feature selection"""
from itertools import chain
from concurrent import futures

import pandas as pd
import numpy as np

import utils

def find_feat_cols(df: pd.DataFrame) -> list:
    """Return list of feature columns"""
    return df.filter(regex="^(?!Metadata_)").columns.to_list()


def find_meta_cols(df: pd.DataFrame) -> list:
    """Return list of metadata columns"""
    return df.filter(regex="^(Metadata_)").columns.to_list()


def remove_nan_infs_columns(input_path: str, output_path: str | None = None):
    """Remove columns with NaN and INF"""
    dframe = pd.read_parquet(input_path)
    feat_cols = find_feat_cols(dframe)
    withnan = dframe[feat_cols].isna().sum()[lambda x: x > 0]
    withinf = (dframe[feat_cols] == np.inf).sum()[lambda x: x > 0]
    withninf = (dframe[feat_cols] == -np.inf).sum()[lambda x: x > 0]
    redlist = set(chain(withinf.index, withnan.index, withninf.index))
    dframe_filtered = dframe[[c for c in dframe.columns if c not in redlist]]
    if output_path is not None:
        dframe_filtered.to_parquet(output_path, index=False)
    else:
        return dframe_filtered


def subtract_well_mean(input_path: str, output_path: str, parallel: bool = True):
    """Subtract the mean of each feature per well position."""
    ann_df = pd.read_parquet(input_path)
    feature_cols = ann_df.filter(regex="^(?!Metadata_)").columns.to_list()

    if parallel:
        def subtract_well_mean_parallel_helper(feature):
            return {
                feature: ann_df[feature] - ann_df.groupby("Metadata_Well")[feature].mean()
            }

        with futures.ThreadPoolExecutor() as executor:
            results = executor.map(subtract_well_mean_parallel_helper, feature_cols)

        for res in results:
            ann_df.update(pd.DataFrame(res))
    else:
        ann_df[feature_cols] = ann_df.groupby("Metadata_Well")[feature_cols].transform(
            lambda x: x - x.mean()
        )

    ann_df.to_parquet(output_path, index=False)
