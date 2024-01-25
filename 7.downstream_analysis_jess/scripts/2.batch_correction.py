'''
Perform batch correction.
'''
import pathlib

from concurrent import futures
import pandas as pd 
from tqdm import tqdm
from sklearn.base import TransformerMixin
from pycytominer.operations.transform import RobustMAD
import time

from utils import get_features, get_metadata

def apply_norm(normalizer: TransformerMixin, df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = get_features(df)
    meta_cols = get_metadata(df)
    meta = df[meta_cols]

    normalizer.fit(df[feat_cols])
    norm_feats = normalizer.transform(df[feat_cols])
    norm_feats = pd.DataFrame(norm_feats,
                              index=df.index,
                              columns=feat_cols)
    df = pd.concat([meta, norm_feats], axis=1)
    return df

def main():
    epsilon_mad = 0.0
    batch_name = 'B1A1R1'

    # Data directories
    data_dir = pathlib.Path("/dgx1nas1/storage/data/jess/varchamp/sc_data/processed_profiles").resolve(strict=True)
    result_dir = pathlib.Path(data_dir / 'batch_corrected/')
    result_dir.mkdir(exist_ok=True)

    # Input file paths
    anno_file = pathlib.Path(data_dir / f"{batch_name}_cc_corrected.parquet")

    # Output file paths
    norm_file = pathlib.Path(result_dir / f"{batch_name}_annotated_corrected_normalized.parquet")

    df = pd.read_parquet(anno_file)
    
    plate_list = list(df['Metadata_Plate'].unique())

    def robust_mad_parallel_helper(plate):
        df_plate = df[df['Metadata_Plate']==plate].copy()
        normalizer = RobustMAD(epsilon_mad)
        df_plate = apply_norm(normalizer, df_plate)
        return df_plate
    
    start = time.perf_counter()

    with futures.ThreadPoolExecutor() as executor:
        results = executor.map(robust_mad_parallel_helper, plate_list)

    end = time.perf_counter()
    print(f'RobustMAD runtime: {end-start} secs.')

    result_list = [res for res in results]
    df_agg = pd.concat(result_list, ignore_index=True)
    df_agg.to_parquet(path=norm_file, compression="gzip", index=False)

if __name__ == '__main__':
    main()