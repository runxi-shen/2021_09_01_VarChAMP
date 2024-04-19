"""Perform correction before feature selection"""
from concurrent import futures
import pandas as pd

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
