"""Helper functions"""

import pandas as pd
import numpy as np
from itertools import chain


def find_feat_cols(df: pd.DataFrame) -> list:
    """Return list of feature columns"""
    return df.filter(regex="^(?!Metadata_)").columns.to_list()


def find_meta_cols(df: pd.DataFrame) -> list:
    """Return list of metadata columns"""
    return df.filter(regex="^(Metadata_)").columns.to_list()

def find_feat_cols_polars(lframe):
    return [col for col in lframe.columns if not col.startswith('Metadata_')]

def find_meta_cols_polars(lframe):
    return [col for col in lframe.columns if col.startswith('Metadata_')]

def remove_nan_infs_columns(dframe: pd.DataFrame):
    """Remove columns with NaN and INF"""
    feat_cols = find_feat_cols(dframe)
    withnan = dframe[feat_cols].isna().sum()[lambda x: x > 0]
    withinf = (dframe[feat_cols] == np.inf).sum()[lambda x: x > 0]
    withninf = (dframe[feat_cols] == -np.inf).sum()[lambda x: x > 0]
    redlist = set(chain(withinf.index, withnan.index, withninf.index))
    dframe_filtered = dframe[[c for c in dframe.columns if c not in redlist]]
    return dframe_filtered
