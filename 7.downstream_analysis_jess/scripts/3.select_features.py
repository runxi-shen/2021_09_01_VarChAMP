import pathlib

import pandas as pd
# ignore mix type warnings from pandas
import warnings
warnings.filterwarnings("ignore")

from pycytominer import feature_select

def main():   
    
    batch_name = 'B6A4R2'
    
    # Data directory
    data_dir = pathlib.Path(f"/dgx1nas1/storage/data/jess/varchamp/sc_data/processed_profiles/{batch_name}").resolve(strict=True)

    # Input path
    norm_path = pathlib.Path(data_dir / f"{batch_name}_annotated_corrected_normalized.parquet")

    # Output path
    feat_path = pathlib.Path(data_dir / f"{batch_name}_annotated_normalized_featselected.parquet")

    # get the allele column
    df = pd.read_parquet(norm_path)

    print('Step 1 starting...')
    # Perform batch level feature selection 
    start_feat = df.shape[1]
    df = feature_select(
                    profiles=df,
                    features="infer",
                    image_features=False,
                    samples="all",
                    operation=["variance_threshold", "blocklist", "drop_na_columns"],
                )
    end_feat = df.shape[1]
    rem_feat = start_feat - end_feat
    print(f'Step 1 completed. {rem_feat} features were removed.')

    # print('Step 2: Noise removal starting...')
    # sample_allele = df['Metadata_allele'].tolist()
    # df_filter_2 = feature_select(
    #                 profiles=df_filter_1,
    #                 features="infer",
    #                 image_features=False,
    #                 samples="all",
    #                 operation="noise_removal",
    #                 noise_removal_perturb_groups = sample_allele,
    #             )
    # print('Step 2: Noise removal completed.')

    print('Step 3: corr_threshold starting...')
    start_feat = df.shape[1]
    df = feature_select(
                    profiles=df,
                    features="infer",
                    image_features=False,
                    samples="all",
                    operation="correlation_threshold",
                )
    end_feat = df.shape[1]
    rem_feat = start_feat - end_feat
    print(f'Step 3 completed. {rem_feat} features were removed.')
    
    df.to_parquet(path=feat_path, compression="gzip", index=False)

if __name__ == "__main__":
    main()