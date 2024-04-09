import pathlib

import pandas as pd
import polars as pl
# ignore mix type warnings from pandas
import warnings
warnings.filterwarnings("ignore")

from pycytominer import feature_select

def main():   
    print("Script started!")

    # Output path
    feat_path = "/dgx1nas1/storage/data/jess/varchamp/sc_data/processed_profiles/Rep_Ctrls/annotated_normalized_featselected.parquet"
    
    # Get unique columns
    lf1 = pl.scan_parquet("/dgx1nas1/storage/data/jess/varchamp/sc_data/processed_profiles/B4A3R1/B4A3R1_annotated_corrected_normalized.parquet")
    lf2 = pl.scan_parquet("/dgx1nas1/storage/data/jess/varchamp/sc_data/processed_profiles/B6A3R2/B6A3R2_annotated_corrected_normalized.parquet")
    
    shared_cols = [value for value in lf1.columns if value in lf2.columns]
    
    lf1 = lf1.select(shared_cols)
    lf2 = lf2.select(shared_cols)
    
    # check to make sure we have the same columns
    lf1.columns == lf2.columns
    
    # concatenate, collect, and covert to pandas
    df = pl.concat([lf1, lf2], how="vertical").collect().to_pandas()

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