"""RobustMAD Normalization"""
from functools import partial
import logging

from scipy.stats import median_abs_deviation

import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from utils import find_feat_cols, find_meta_cols, remove_nan_infs_columns

logger = logging.getLogger(__name__)


def split_parquet(
    dframe_path, features=None
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
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


def add_metadata(stats: pd.DataFrame, meta: pd.DataFrame):
    """Add metadata to plate statistics"""
    source_map = meta[["Metadata_Source", "Metadata_Plate"]].drop_duplicates()
    source_map = source_map.set_index("Metadata_Plate").Metadata_Source
    stats["Metadata_Source"] = stats["Metadata_Plate"].map(source_map)
    parts = stats["feature"].str.split("_", expand=True)
    stats["compartment"] = parts[0].astype("category")
    stats["family"] = parts[range(3)].apply("_".join, axis=1).astype("category")


def get_plate_stats(input_path: str, output_path: str):
    """Compute plate level statistics"""
    mad_fn = partial(median_abs_deviation, nan_policy="omit", axis=0)
    dframe = pd.read_parquet(input_path)

    dframe = remove_nan_infs_columns(dframe)
    meta_cols = find_meta_cols(dframe)
    feat_cols = find_feat_cols(dframe)
    dframe = dframe[feat_cols + ["Metadata_Plate"]]
    median = dframe.groupby("Metadata_Plate", observed=True).median()
    max_ = dframe.groupby("Metadata_Plate", observed=True).max()
    min_ = dframe.groupby("Metadata_Plate", observed=True).min()
    count = dframe.groupby("Metadata_Plate", observed=True).count()
    mad = dframe.groupby("Metadata_Plate", observed=True).apply(mad_fn)
    mad = pd.DataFrame(index=mad.index, data=np.stack(mad.values), columns=feat_cols)

    median["stat"] = "median"
    mad["stat"] = "mad"
    min_["stat"] = "min"
    max_["stat"] = "max"
    count["stat"] = "count"

    stats = pd.concat([median, mad, min_, max_, count])
    stats.reset_index(inplace=True)
    stats = stats.melt(id_vars=["Metadata_Plate", "stat"], var_name="feature")
    stats = stats.pivot(
        index=["Metadata_Plate", "feature"], columns="stat", values="value"
    )
    stats.reset_index(inplace=True)
    stats["abs_coef_var"] = (
        (stats["mad"] / stats["median"]).fillna(0).abs().replace(np.inf, 0)
    )
    stats = stats.astype(
        {
            "min": np.float32,
            "max": np.float32,
            "count": np.int32,
            "median": np.float32,
            "mad": np.float32,
            "abs_coef_var": np.float32,
            "feature": "category",
        }
    )
    add_metadata(stats, dframe[meta_cols])
    stats.to_parquet(output_path)


def select_variant_features(parquet_path, norm_stats_path, variant_feats_path):
    """
    Filtered out features that have mad == 0 or abs_coef_var>1e-3 in any plate.
    stats are computed using negative controls only
    """
    dframe = pd.read_parquet(parquet_path)
    norm_stats = pd.read_parquet(norm_stats_path)

    # Remove NaN and Inf
    dframe = remove_nan_infs_columns(dframe)

    # Select variant_features
    norm_stats = norm_stats.query("mad!=0 and abs_coef_var>1e-3")
    groups = norm_stats.groupby("Metadata_Plate", observed=True)["feature"]
    variant_features = set.intersection(*groups.agg(set).tolist())

    # Select plates with variant features
    norm_stats = norm_stats.query("feature in @variant_features")
    dframe = dframe.query("Metadata_Plate in @norm_stats.Metadata_Plate")

    # Filter features
    variant_features = sorted(variant_features)
    meta = dframe[find_meta_cols(dframe)]
    vals = dframe[variant_features].values
    merge_parquet(meta, vals, variant_features, variant_feats_path)


def robustmad(variant_feats_path, plate_stats_path, normalized_path):
    """Perform plate level RobustMAD (value - median)/mad"""
    meta, vals, features = split_parquet(variant_feats_path)
    plate_stats = pd.read_parquet(plate_stats_path)
    plate_stats = plate_stats.query("feature in @features")

    # get counts and sort by plate
    plates, counts = np.unique(meta["Metadata_Plate"], return_counts=True)
    ix = np.argsort(meta["Metadata_Plate"])
    meta = meta.iloc[ix]
    vals = vals[ix]

    # get mad and median matrices for MAD normalization
    mads = plate_stats.pivot(index="Metadata_Plate", columns="feature", values="mad")
    mads = mads.loc[plates, features].values
    medians = plate_stats.pivot(
        index="Metadata_Plate", columns="feature", values="median"
    )
    medians = medians.loc[plates, features].values

    # Get normalized features (epsilon = 0) for all plates that have MAD stats
    # -= and /= are inplace operations. i.e save memory
    vals -= np.repeat(medians, counts, axis=0)
    vals /= np.repeat(mads, counts, axis=0)

    merge_parquet(meta, vals, features, normalized_path)
