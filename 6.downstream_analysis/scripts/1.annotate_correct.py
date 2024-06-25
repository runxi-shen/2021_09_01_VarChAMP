'''
Annotate and perform well-postion and cell count regression.
'''
import os
import pathlib
from typing import Optional

from concurrent import futures
import pandas as pd
from statsmodels.formula.api import ols
from tqdm import tqdm
import time

from utils import get_features

def annotate_with_platemap(profile_path: str, platemap_path: str, output_file_path: str | None = None):
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

def subtract_well_mean(ann_df: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract the mean of each feature per each well.

    Parameters
    ----------
    ann_df : pandas.DataFrame
        Dataframe with features and metadata.

    Returns
    -------
    pandas.DataFrame
        Dataframe with features and metadata, with each feature subtracted by the mean of that feature per well.
    """
    feature_cols = ann_df.filter(regex="^(?!Metadata_)").columns.to_list()

    def subtract_well_mean_parallel_helper(feature):
        return {feature: ann_df[feature] - ann_df.groupby("Metadata_Well")[feature].mean()}
    
    with futures.ThreadPoolExecutor() as executor:
        results = executor.map(subtract_well_mean_parallel_helper, feature_cols)

    for res in results:
        ann_df.update(pd.DataFrame(res))

    return ann_df

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
    
def main():
    """Annotate and aggregate plate-level profiles.
    """
    
    # Input directories
    data_dir = pathlib.Path("/dgx1nas1/storage/data/sam/profiles").resolve(strict=True)
    result_dir = pathlib.Path("/dgx1nas1/storage/data/sam/processed_v2")
    result_dir.mkdir(exist_ok=True)

    # Output file paths
    batch_name = '2023_05_30_B1A1R1'
    anot_file = pathlib.Path(result_dir / f'{batch_name}_annotated.parquet')
    well_corrected_file = (result_dir / f'{batch_name}_well_corrected.parquet')
    cc_file = pathlib.Path(result_dir / f'{batch_name}_cc_corrected.parquet')
    # norm_file = pathlib.Path(result_dir / batch_name + '_annotated_normalized.parquet')

    # Platemap
    platemap_file = '/dgx1nas1/storage/data/sam/codes/2021_09_01_VarChAMP/6.downstream_analysis/2023_05_30_B1A1R1.csv'
    df_well_path = f'{data_dir}/VarChamp/Well_level/2023_05_30_B1A1R1_well_level.parquet'
    
    # Annotate profiles
    plate_list = []
    for file in tqdm(os.listdir(data_dir)):
        if not file.endswith(".parquet"): continue
        orig_file = pathlib.Path(data_dir / file).resolve(strict=True)
        df_ann = annotate_with_platemap(orig_file, platemap_file)
        plate_list.append(df_ann)

    # # Aggregate profiles and save
    df_agg = pd.concat(plate_list, ignore_index=True)
    df_agg.to_parquet(path=anot_file, compression="gzip", index=False)
    
    # Well position correction

    start = time.perf_counter()
    df_agg = subtract_well_mean(df_agg)
    end = time.perf_counter()
    print(f'Well position correction runtime: {end-start}.')
    df_agg.to_parquet(path=well_corrected_file, compression="gzip", index=False)
    
    # Cell count regression
    df_agg = pd.read_parquet(well_corrected_file)

    start = time.perf_counter()
    df_well = pd.read_parquet(df_well_path)
    df_agg = regress_out_cell_counts(df_well, df_agg,'Metadata_Object_Count')
    end = time.perf_counter()
    print(f'Cell count regression runtime: {end-start}.')
    df_agg.to_parquet(path=cc_file, compression="gzip", index=False)
    print(f"\n Position and cell count corrected profiles saved in: {cc_file}")

    # Whole plate normalization
    # normalize(
    #     profiles=anot_file,
    #     features="infer",
    #     image_features=False,
    #     meta_features="infer",
    #     samples="all",
    #     method='mad_robustize',
    #     mad_robustize_epsilon=0,
    #     output_file=norm_file,
    #     output_type="parquet",
    #     compression_options="gzip",
    # )
    
if __name__ == "__main__":
    main()