'''
Perform batch correction.
'''
import pathlib

import pandas as pd 
from tqdm import tqdm
from sklearn.base import TransformerMixin
from pycytominer.operations.transform import RobustMAD

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
    batch_name = '2023_05_30_B1A1R1'

    # Data directories
    data_dir = pathlib.Path("/dgx1nas1/storage/data/sam/processed_v2").resolve(strict=True)
    result_dir = pathlib.Path(data_dir / 'batch_corrected/')
    result_dir.mkdir(exist_ok=True)

    # Input file paths
    anno_file = pathlib.Path(data_dir / batch_name + '_cc_corrected.parquet')

    # Output file paths
    norm_file = pathlib.Path(result_dir / batch_name + '_annotated_corrected_normalized.parquet')

    df = pd.read_parquet(anno_file)
    
    norm_plates = []
    plate_list = df['Metadata_Plate'].unique().to_list()
    for plate in tqdm(plate_list):
        df_plate = df[df['Metadata_Plate']==plate].copy()

        normalizer = RobustMAD(epsilon_mad)
        df_plate = apply_norm(normalizer, df_plate)
        norm_plates.append(df_plate)

    df_agg = pd.concat(norm_plates, index=False)
    df_agg.to_parquet(path=norm_file, compression="gzip", index=False)

if __name__ == '__main__':
    main()