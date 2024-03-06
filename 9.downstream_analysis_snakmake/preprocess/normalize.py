'''RobustMAD Normalization'''
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

from .correct import find_feat_cols, find_meta_cols, remove_nan_infs_columns

def split_parquet(dframe_path,
                  features=None) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Split dframe into metadata, feature values, feature columns"""
    dframe = pd.read_parquet(dframe_path)
    if features is None:
        features = find_feat_cols(dframe)
    vals = np.empty((len(dframe), len(features)), dtype=np.float32)
    for i, c in enumerate(features):
        vals[:, i] = dframe[c]
    meta = dframe[find_meta_cols(dframe)].copy()
    return meta, vals, features

def merge_parquet(meta, vals, features, output_path) -> None:
    """Save the data in a parquet file resetting the index"""
    dframe = pd.DataFrame(vals, columns=features)
    for c in meta:
        dframe[c] = meta[c].reset_index(drop=True)
    dframe.to_parquet(output_path)

def compute_negcon_stats(parquet_path: str, neg_stats_path: str, negcon_query: str):
    """create statistics of negative controls platewise
      for columns without nan/inf values only"""
    logger.info('Loading data')
    dframe = pd.read_parquet(parquet_path)
    logger.info('Removing nan and inf columns')
    dframe = remove_nan_infs_columns(dframe)
    negcon = dframe.query('Metadata_JCP2022 == "DMSO"')
    logger.info('computing stats for negcons')
    neg_stats = get_plate_stats(negcon)
    logger.info('stats done.')
    add_metadata(neg_stats, dframe[find_meta_cols(dframe)])
    neg_stats.to_parquet(neg_stats_path)

def mad(variant_feats_path, neg_stats_path, normalized_path):
    meta, vals, features = split_parquet(variant_feats_path)
    neg_stats = pd.read_parquet(neg_stats_path)
    neg_stats = neg_stats.query('feature in @features')

    # get counts and sort by plate
    plates, counts = np.unique(meta['Metadata_Plate'], return_counts=True)
    ix = np.argsort(meta['Metadata_Plate'])
    meta = meta.iloc[ix]
    vals = vals[ix]

    # get mad and median matrices for MAD normalization
    mads = neg_stats.pivot(index='Metadata_Plate',
                           columns='feature',
                           values='mad')
    mads = mads.loc[plates, features].values
    medians = neg_stats.pivot(index='Metadata_Plate',
                              columns='feature',
                              values='median')
    medians = medians.loc[plates, features].values

    # Get normalized features (epsilon = 0) for all plates that have MAD stats
    # -= and /= are inplace operations. i.e save memory
    vals -= np.repeat(medians, counts, axis=0)
    vals /= np.repeat(mads, counts, axis=0)

    merge_parquet(meta, vals, features, normalized_path)