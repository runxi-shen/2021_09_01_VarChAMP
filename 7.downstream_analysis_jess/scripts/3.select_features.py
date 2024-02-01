import pathlib

import pandas as pd
from tqdm import tqdm
# ignore mix type warnings from pandas
import warnings
warnings.filterwarnings("ignore")

from pycytominer import feature_select

def main():   
    # Data directory
    data_dir = pathlib.Path("/dgx1nas1/storage/data/jess/varchamp/sc_data/processed_profiles").resolve(strict=True)
    result_dir = data_dir

    # Input path
    batch_name = 'B1A1R1'
    norm_path = pathlib.Path(data_dir / f"{batch_name}_annotated_corrected_normalized.parquet")

    # Output path
    feat_path_1 = pathlib.Path(result_dir / f"{batch_name}_annotated_normalized_feat_selected_1.parquet")
    feat_path_2 = pathlib.Path(result_dir / f"{batch_name}_annotated_normalized_feat_selected_2.parquet")
    feat_path_3 = pathlib.Path(result_dir / f"{batch_name}_annotated_normalized_feat_selected_3.parquet")

    # get the allele column
    df = pd.read_parquet(norm_path)
    sample_allele = df['Metadata_allele'].tolist()

    print('Step 1 starting...')
    # Perform batch level feature selection 
    df_filter_1 = feature_select(
                    profiles=df,
                    features="infer",
                    image_features=False,
                    samples="all",
                    operation=["variance_threshold", "blocklist", "drop_na_columns"],
                )
    df_filter_1.to_parquet(path=feat_path_1, compression="gzip", index=False)
    print('Step 1 completed.')

    print('Step 2: Noise removal starting...')
    df_filter_2 = feature_select(
                    profiles=df_filter_1,
                    features="infer",
                    image_features=False,
                    samples="all",
                    operation="noise_removal",
                    noise_removal_perturb_groups = sample_allele,
                )
    df_filter_2.to_parquet(path=feat_path_2, compression="gzip", index=False)
    print('Step 2: Noise removal completed.')

    print('Step 3: corr_threshold starting...')
    df_filter_3 = feature_select(
                    profiles=df_filter_2,
                    features="infer",
                    image_features=False,
                    samples="all",
                    operation="correlation_threshold",
                )
    df_filter_3.to_parquet(path=feat_path_3, compression="gzip", index=False)
    print('Step 3: corr_threshold completed.')

if __name__ == "__main__":
    main()