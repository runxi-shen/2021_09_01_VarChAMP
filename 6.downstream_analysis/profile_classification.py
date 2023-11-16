import pandas as pd
import numpy as np
import pathlib
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score, classification_report
# from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import xgboost as xgb
import random

def classifier(all_profiles, feat_col, target='Label', stratify_label=None):
    X, y = all_profiles[feat_col], all_profiles[[target]]
    if not (stratify_label is None):
        stratify = all_profiles[stratify_label]
    else: stratify = y
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=stratify
    )
    # Model train and predict
    model = xgb.XGBClassifier().fit(X_train, y_train)
    # model = LogisticRegression().fit(X_train, y_train)
    preds = model.predict(X_test)

    # Store feature importance
    # feat_importances = pd.Series(
    #     model.feature_importances_, index=X_train.columns
    # )
    feat_importances = pd.Series(
        permutation_importance(model, X, y)['importances_mean'], index=X_train.columns
    )
    # Evaluate with metrics
    f1score_macro = f1_score(y_test, preds, average="macro")
      
    return feat_importances, f1score_macro

def experimental_group_runner(sc_profiles, gene_group, data_dir, feat_col, batch_name='', 
    protein_prefix='protein', feature_type='_normalized_feature_selected'):
    gene_list = []
    pair_list = []
    feat_list = []
    f1score_macro_list = []

    for gene_key in tqdm(gene_group.keys()):
        gene_profiles = sc_profiles.loc[gene_group[gene_key]]

        # Ensure this gene has both reference and variants
        if gene_profiles.Metadata_node_type.unique().size != 2:
            continue

        # All wildtype cells for the gene
        ref_profiles = gene_profiles[
            gene_profiles["Metadata_node_type"] == "disease_wt"
        ].reset_index(drop=True)

        ref_plate_list = list(ref_profiles['Metadata_Plate'].unique())
        
        var_group = (
            gene_profiles[gene_profiles["Metadata_node_type"] == "allele"]
            .groupby("Metadata_Variant")
            .groups
        )

        for var_key in var_group.keys():
            # All cells from the specific variant
            var_profiles = gene_profiles.loc[var_group[var_key]]
            var_plate_list = list(var_profiles['Metadata_Plate'].unique())

            ref_profiles["Label"] = 1
            var_profiles["Label"] = 0
            
            plate_list = list(set(ref_plate_list) & set(var_plate_list))
            unique_ref_plate_list = list(set(ref_plate_list) - set(var_plate_list))
            unique_var_plate_list = list(set(var_plate_list) - set(ref_plate_list))
            
            for plate in plate_list:
                ref_plate_profiles = ref_profiles[ref_profiles['Metadata_Plate']==plate]
                var_plate_profiles = var_profiles[var_profiles['Metadata_Plate']==plate]

                all_profiles = pd.concat(
                [ref_plate_profiles, var_plate_profiles], ignore_index=True
                )

                feat_importances, f1score_macro = classifier(all_profiles, feat_col)
        
                feat_list.append(feat_importances)
                f1score_macro_list.append(f1score_macro)
                gene_list.append(gene_key)
                pair_list.append(var_key)

            if len(plate_list) == 0:
                for i in range(min(len(unique_ref_plate_list), len(unique_var_plate_list))):
                    ref_plate_profiles = ref_profiles[ref_profiles['Metadata_Plate']==unique_ref_plate_list[i]]
                    var_plate_profiles = var_profiles[var_profiles['Metadata_Plate']==unique_var_plate_list[i]]
    
                    all_profiles = pd.concat(
                    [ref_plate_profiles, var_plate_profiles], ignore_index=True
                    )
    
                    feat_importances, f1score_macro = classifier(all_profiles, feat_col, stratify_label='Metadata_Plate')
            
                    feat_list.append(feat_importances)
                    f1score_macro_list.append(f1score_macro)
                    gene_list.append(gene_key)
                    pair_list.append(var_key)

    df_feat_one = pd.DataFrame({"Gene": gene_list, "Variant": pair_list})
    df_feat_two = pd.DataFrame(feat_list)
    df_feat = pd.concat([df_feat_one, df_feat_two], axis=1)

    df_feat.to_csv(
    f"{data_dir}/{batch_name}_{protein_prefix}_feat_importance{feature_type}.csv",
    index=False,
)

    result_csv = pd.DataFrame(
    {
        "Gene": gene_list,
        "Variant": pair_list,
        "F1_Score": f1score_macro_list
    }
)
    result_csv.to_csv(
        f"{data_dir}/{batch_name}_{protein_prefix}_f1score{feature_type}.csv",
        index=False,
    )

def control_group_runner(controls, control_group, data_dir, feat_col, batch_name='', 
    protein_prefix='protein', feature_type='_normalized_feature_selected'):
    f1score_macro_list = []
    gene_list = []
    pair_list = []
    feat_list = []
    for gene_key in tqdm(control_group.keys()):
        gene_profiles = controls.loc[control_group[gene_key]]
        # Skip controls with no replicates
        if gene_profiles.Metadata_Well.unique().size < 2:
            continue
        
        gene_profiles['Plate_Well'] = gene_profiles['Metadata_Plate'] + '_' + gene_profiles['Metadata_Well']
        plate_group = gene_profiles.groupby("Metadata_Plate").groups
        
        for plate in plate_group.keys():
            plate_profiles = gene_profiles.loc[plate_group[plate]]
            well_group = plate_profiles.groupby("Metadata_Well").groups
            well_list = list(well_group.keys())
    
            for i in range(len(well_list) - 1):
                # All cells from the specific variant
                well_one = plate_profiles.loc[well_group[well_list[i]]]
                well_one["Label"] = 1
    
                for j in range(i + 1, len(well_list)):
                    well_two = plate_profiles.loc[well_group[well_list[j]]]
                    well_two["Label"] = 0
    
                    all_profiles = pd.concat([well_one, well_two], ignore_index=True)
                    feat_importances, f1score_macro = classifier(all_profiles, feat_col)
    
                    feat_list.append(feat_importances)
                    f1score_macro_list.append(f1score_macro)
                    gene_list.append(gene_key)
                    pair_list.append(plate + '_' + well_list[i] + "_" + well_list[j])

    df_feat_one = pd.DataFrame({"Gene": gene_list, "Variant": pair_list})
    df_feat_two = pd.DataFrame(feat_list)
    df_feat = pd.concat([df_feat_one, df_feat_two], axis=1)

    df_feat.to_csv(
    f"{data_dir}/{batch_name}_{protein_prefix}_control_feat_importance{feature_type}.csv",
    index=False,
)

    result_csv = pd.DataFrame(
    {
        "Treatment": gene_list,
        "Well_Pair": pair_list,
        "F1_Score": f1score_macro_list
    }
)
    result_csv.to_csv(
        f"{data_dir}/{batch_name}_{protein_prefix}_control_f1score{feature_type}.csv",
        index=False,
    )

def main():
    data_dir = "/dgx1nas1/storage/data/sam/processed"
    feature_type = "_normalized_feature_selected"
    batch = "2023_05_30_B1A1R1"
    run_name = 'Run4_plate_annot'
    
    drop_control_feat = False
    percent_dropping = 0.1
    feat_rank_dir = '/dgx1nas1/storage/data/sam/results/Run6'
    
    result_dir = pathlib.Path(f'/dgx1nas1/storage/data/sam/results/{run_name}')
    result_dir.mkdir(exist_ok=True)

    sc_profile_path = f"{data_dir}/{batch}{feature_type}.parquet"
    sc_profiles = pd.read_parquet(sc_profile_path)

    # sc_profiles.drop(["ObjectNumber", "ObjectNumber_Cells"], axis=1, inplace=True)
    meta_col = [i for i in sc_profiles.columns if "Metadata_" in i]
    feat_col = [i for i in sc_profiles.columns if "Metadata_" not in i]

    # Drop features that contributed to high performance in null
    if drop_control_feat:
        df_protein_feat_rank = pd.read_csv(f'{feat_rank_dir}/ctrl_protein_feat_rank.csv')
        df_non_protein_feat_rank = pd.read_csv(f'{feat_rank_dir}/ctrl_non_protein_feat_rank.csv')

        df_protein_drop = list(df_protein_feat_rank['feature'][0:int(df_protein_feat_rank.shape[0]*percent_dropping)])
        df_non_protein_drop = list(df_non_protein_feat_rank['feature'][0:int(df_non_protein_feat_rank.shape[0]*percent_dropping)])
        sc_profiles.drop(df_protein_drop+df_non_protein_drop, axis=1, inplace=True)
        print(f'Removed {len(df_protein_drop+df_non_protein_drop)} features that dominated control predictions.')
        
    # Drop rows that failed to merge with metadata
    row_count = int(sc_profiles.shape[0])
    sc_profiles.drop(np.where(sc_profiles['Metadata_Batch'].isna())[0], axis=0, inplace=True)
    sc_profiles.reset_index(drop=True, inplace=True)
    print(f'Removed {row_count-sc_profiles.shape[0]} rows with NaN metadata values.')

    # Drop features with null values
    r,c = np.where(sc_profiles.isna())
    feat, count = np.unique(c, return_counts = True)
    
    feat_to_remove = []
    row_to_remove = []
    cell_threshold = 100
    
    for i in range(len(feat)):
        feat_name = sc_profiles.columns[feat[i]]
        if feat_name.startswith('Metadata_'): continue
            
        if count[i]>cell_threshold:
            feat_to_remove.append(feat_name)
        else:
            row_to_remove = row_to_remove + np.where(sc_profiles[feat_name].isna())[0].tolist()
            row_to_remove = list(set(row_to_remove))
        
    # Drop features with null values
    # r,c = np.where(sc_profiles.isna())
    # features_to_remove = [_ for _ in list(sc_profiles.columns[list(set(c))]) if not _.startswith('Metadata_')]
    sc_profiles.drop(feat_to_remove, axis=1, inplace=True)
    sc_profiles.drop(row_to_remove, axis=0, inplace=True)
    sc_profiles.reset_index(drop=True, inplace=True)
    print(f'Removed {len(feat_to_remove)} nan features and {len(row_to_remove)} nan rows.')

    feat_col = [i for i in sc_profiles.columns if "Metadata_" not in i]
    assert ~np.isnan(sc_profiles[feat_col]).any().any(), "Dataframe contain NaN features." 
    assert np.isfinite(sc_profiles[feat_col]).all().all(), "Dataframe contain infinite feature values."
    
    # Include only GFP features for protein channel 
    feat_cols_protein = [
        i
        for i in feat_col
        if ("GFP" in i)
        and ("DNA" not in i)
        and ("AGP" not in i)
        and ("Mito" not in i)
    ]
    # Select non-protein channel features, where GFP does not exist in feat_cols
    feat_cols_non_protein = [i for i in feat_col if "GFP" not in i]

    experiments = sc_profiles[~sc_profiles["Metadata_control"].astype('bool')].reset_index(drop=True)
    gene_group = experiments.groupby("Metadata_Gene").groups

    controls = sc_profiles[sc_profiles["Metadata_control"].astype('bool')].reset_index(drop=True)
    control_group = controls.groupby("Metadata_Sample_Unique").groups

    control_group_runner(
        controls, 
        control_group, 
        result_dir, 
        feat_cols_protein, 
        batch_name=batch,
        protein_prefix='protein')
    print('Finished protein feature classification for control groups.\n')

    experimental_group_runner(
        experiments, 
        gene_group, 
        result_dir, 
        feat_cols_protein, 
        batch_name=batch,
        protein_prefix='protein')
    print('Finished protein feature classification for experimental groups.\n')

    control_group_runner(
        controls, 
        control_group, 
        result_dir, 
        feat_cols_non_protein, 
        batch_name=batch,
        protein_prefix='non_protein')
    print('Finished non-protein feature classification for control groups.\n')
    
    
    experimental_group_runner(
        experiments, 
        gene_group, 
        result_dir, 
        feat_cols_non_protein, 
        batch_name=batch,
        protein_prefix='non_protein')
    print('Finished non-protein feature classification for experimental groups.\n')

    

main()
            