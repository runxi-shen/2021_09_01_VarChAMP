import pandas as pd
import numpy as np
import pathlib
from tqdm import tqdm
import os

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.inspection import permutation_importance
import xgboost as xgb
import random
from itertools import combinations

def drop_top_control_feat(sc_profiles, feat_rank_dir, percent_dropping=0.1):
    '''
    This function drops the features with highest weight in previous run of control experiments. 
    '''
    df_protein_feat_rank = pd.read_csv(f'{feat_rank_dir}/ctrl_protein_feat_rank.csv')
    df_non_protein_feat_rank = pd.read_csv(f'{feat_rank_dir}/ctrl_non_protein_feat_rank.csv')

    df_protein_drop = list(df_protein_feat_rank['feature'][0:int(df_protein_feat_rank.shape[0]*percent_dropping)])
    df_non_protein_drop = list(df_non_protein_feat_rank['feature'][0:int(df_non_protein_feat_rank.shape[0]*percent_dropping)])
    sc_profiles.drop(df_protein_drop+df_non_protein_drop, axis=1, inplace=True)
    print(f'Removed {len(df_protein_drop+df_non_protein_drop)} features that dominated control predictions.')
    return sc_profiles

def drop_meta_null(sc_profiles, check_col='Metadata_Batch'):
    '''
    This function drops the rows that contain null value in metadata (failure in merging). 
    '''
    row_count = int(sc_profiles.shape[0])
    sc_profiles.drop(np.where(sc_profiles[check_col].isna())[0], axis=0, inplace=True)
    sc_profiles.reset_index(drop=True, inplace=True)
    print(f'Removed {row_count-sc_profiles.shape[0]} rows with NaN metadata values.')
    return sc_profiles

def drop_null_features(sc_profiles, cell_threshold=100):
    '''
    This function first drops the features that contain more than 100 rows of null cells, 
    then drop the rest of null rows. 
    '''
    # Drop features with null values
    r,c = np.where(sc_profiles.isna())
    feat, count = np.unique(c, return_counts = True)
    
    feat_to_remove = []
    row_to_remove = []
    
    for i in range(len(feat)):
        feat_name = sc_profiles.columns[feat[i]]
        if feat_name.startswith('Metadata_'): continue

        # If more than @cell_threshold number of cells has null for one feature, drop that feature
        # else remove the null row
        if count[i]>cell_threshold:
            feat_to_remove.append(feat_name)
        else:
            row_to_remove = row_to_remove + np.where(sc_profiles[feat_name].isna())[0].tolist()
            row_to_remove = list(set(row_to_remove))
        
    sc_profiles.drop(feat_to_remove, axis=1, inplace=True)
    sc_profiles.drop(row_to_remove, axis=0, inplace=True)
    sc_profiles.reset_index(drop=True, inplace=True)
    print(f'Removed {len(feat_to_remove)} nan features and {len(row_to_remove)} nan rows.')
    feat_col = [i for i in sc_profiles.columns if "Metadata_" not in i]

    # Ensure no null rows or columns
    assert ~np.isnan(sc_profiles[feat_col]).any().any(), "Dataframe contain NaN features." 
    assert np.isfinite(sc_profiles[feat_col]).all().all(), "Dataframe contain infinite feature values."
    return sc_profiles

def classifier(all_profiles_train, all_profiles_test, feat_col, target='Label', stratify_label=None, eval=False):
    '''
    This function runs classification.
    '''
    if eval:
        X_train, X_validation, y_train, y_validation = train_test_split(
            all_profiles_train[feat_col], all_profiles_train[[target]], test_size=0.2, random_state=1
        )
        eval_set = [(X_train, y_train), (X_validation, y_validation)]
        
        X_test, y_test = all_profiles_test[feat_col], all_profiles_test[[target]]

        model = xgb.XGBClassifier(objective='binary:logistic',
                            n_estimators=300,
                            tree_method='hist',
                            learning_rate = 0.05,
                            early_stopping_rounds = 100,
                            device="cuda",
                            verbosity=0)
        
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        
    else: 
        X_train, y_train = all_profiles_train[feat_col], all_profiles_train[[target]]
        X_test, y_test = all_profiles_test[feat_col], all_profiles_test[[target]]
        
        model = xgb.XGBClassifier(objective='binary:logistic',
                            n_estimators=150,
                            tree_method='hist',
                            learning_rate = 0.05,
                            device="cuda:7").fit(X_train, y_train, verbose=False)

    # model = LogisticRegression().fit(X_train, y_train)
    preds = model.predict(X_test)

    # Store feature importance
    feat_importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    )
    
    # Evaluate with metrics
    f1score_macro = f1_score(y_test, preds, average="macro")
      
    return feat_importances, f1score_macro

def experimental_group_runner(experiments, gene_group, data_dir, feat_col, batch_name='', 
    protein_prefix='protein', feature_type='_normalized_feature_selected'):
    '''
    This function runs the reference v.s. variants experiments. 
    '''
    gene_list = []
    pair_list = []
    feat_list = []
    f1score_macro_list = []
    result_plate_list = []

    for gene_key in tqdm(gene_group.keys()):
        gene_profiles = experiments.loc[gene_group[gene_key]]

        # Ensure this gene has both reference and variants
        if gene_profiles.Metadata_node_type.unique().size != 2:
            continue

        # All wildtype cells for the gene
        ref_profiles = gene_profiles[
            gene_profiles["Metadata_node_type"] == "disease_wt"
        ].reset_index(drop=True)        
        ref_profiles["Label"] = 1
        
        ref_plate_list = list(ref_profiles['Metadata_Plate'].unique())
        
        var_group = (
            gene_profiles[gene_profiles["Metadata_node_type"] == "allele"]
            .groupby("Metadata_Variant")
            .groups
        )
        
        for var_key in var_group.keys():
            # All cells from one variant
            var_profiles = gene_profiles.loc[var_group[var_key]].reset_index(drop=True)
            var_profiles["Label"] = 0
            var_plate_list = list(var_profiles['Metadata_Plate'].unique()) 

            # Only compare profiles on the same plate
            plate_list = list(set(ref_plate_list) & set(var_plate_list))
            
            for plate in plate_list:
                # Train on data from other plates
                all_profiles_train = pd.concat(
                [ref_profiles[ref_profiles['Metadata_Plate']!=plate], 
                 var_profiles[var_profiles['Metadata_Plate']!=plate]], 
                ignore_index=True
                )
                
                # Test on data from one plate
                all_profiles_test = pd.concat(
                [ref_profiles[ref_profiles['Metadata_Plate']==plate], 
                 var_profiles[var_profiles['Metadata_Plate']==plate]], 
                ignore_index=True
                )
                feat_importances, f1score_macro = classifier(all_profiles_train, all_profiles_test, feat_col)
                
                feat_list.append(feat_importances)
                f1score_macro_list.append(f1score_macro)
                gene_list.append(gene_key)
                pair_list.append(var_key)
                result_plate_list.append(plate)

    df_feat_one = pd.DataFrame({"Gene": gene_list, "Variant": pair_list})
    df_feat_two = pd.DataFrame(feat_list)
    df_feat = pd.concat([df_feat_one, df_feat_two], axis=1)

    df_feat.to_csv(f"{data_dir}/{batch_name}_{protein_prefix}_feat_importance{feature_type}.csv", index=False)
    
    result_csv = pd.DataFrame(
    {
        "Gene": gene_list,
        "Variant": pair_list,
        "Plate": result_plate_list,
        "F1_Score": f1score_macro_list
    }
    )
    result_csv.to_csv(f"{data_dir}/{batch_name}_{protein_prefix}_f1score{feature_type}.csv",index=False)

def control_group_runner(controls, control_group, data_dir, feat_col, batch_name='', 
    protein_prefix='protein', feature_type='_normalized_feature_selected'):
    '''
    This function runs the null control experiments. 
    '''
    f1score_macro_list = []
    gene_list = []
    pair_list = []
    feat_list = []
    result_plate_list = []
    
    for gene_key in tqdm(control_group.keys()):
        gene_profiles = controls.loc[control_group[gene_key]]
        # Skip controls with no replicates
        if gene_profiles.Metadata_Well.unique().size < 2:
            continue
            
        well_group = gene_profiles.groupby("Metadata_Well").groups
        
        # Randomly choose 4 out of 6 possible combinations of well pairs
        num_pair = 4
        well_pair_list = random.choices([ x for x in combinations(list(well_group.keys()), r=2)], k=num_pair)

        for (idx_one, idx_two) in well_pair_list:
            well_one = gene_profiles.loc[well_group[idx_one]].reset_index(drop=True)
            well_one["Label"] = 1
            well_two = gene_profiles.loc[well_group[idx_two]].reset_index(drop=True)
            well_two["Label"] = 0

            plate_list = list(set(list(well_one['Metadata_Plate'].unique())) 
                              & set(list(well_two['Metadata_Plate'].unique()))
                             )
            
            for plate in plate_list:
                platemap = plate.split('_')[-1].split('T')[0]
                
                # Train on data from same platemap but other plates
                all_profiles_train = pd.concat(
                [well_one[well_one['Metadata_Plate'].apply(lambda x: (platemap in x) & (plate not in x))], 
                 well_two[well_two['Metadata_Plate'].apply(lambda x: (platemap in x) & (plate not in x))]],
                ignore_index=True
                )
                
                # Test on data from one replicate plate
                all_profiles_test = pd.concat(
                [well_one[well_one['Metadata_Plate']==plate], 
                 well_two[well_two['Metadata_Plate']==plate]], 
                ignore_index=True
                )
                
                feat_importances, f1score_macro = classifier(all_profiles_train, all_profiles_test, feat_col)

                feat_list.append(feat_importances)
                f1score_macro_list.append(f1score_macro)
                gene_list.append(gene_key)
                pair_list.append(idx_one + "_" + idx_two)
                result_plate_list.append(plate)
        
    df_feat_one = pd.DataFrame({"Gene": gene_list, "Variant": pair_list})
    df_feat_two = pd.DataFrame(feat_list)
    df_feat = pd.concat([df_feat_one, df_feat_two], axis=1)

    df_feat.to_csv(f"{data_dir}/{batch_name}_{protein_prefix}_control_feat_importance{feature_type}.csv",index=False)

    result_csv = pd.DataFrame(
    {
        "Treatment": gene_list,
        "Well_Pair": pair_list,
        "Plate": result_plate_list,
        "F1_Score": f1score_macro_list
    }
)
    result_csv.to_csv(f"{data_dir}/{batch_name}_{protein_prefix}_control_f1score{feature_type}.csv",index=False)
    

    
def main():
    data_dir = "/dgx1nas1/storage/data/jess/varchamp/sc_data/processed_profiles"
    feature_type = "_annotated_normalized_feat_selected_3"
    batch = "B1A1R1"
    run_name = 'Run1'
    os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
    
    drop_control_feat = False
    percent_dropping = 0.1
    feat_rank_dir = pathlib.Path(f'/dgx1nas1/storage/data/jess/varchamp/sc_data/classification_results/{run_name}')
    
    result_dir = pathlib.Path(f'/dgx1nas1/storage/data/jess/varchamp/sc_data/classification_results/{run_name}')
    result_dir.mkdir(exist_ok=True)

    sc_profile_path = f"{data_dir}/{batch}{feature_type}.parquet"
    sc_profile_cleaned_path = f"{data_dir}/{batch}{feature_type}_nonull.parquet"

    # Remove null values in dataframe 
    if not os.path.exists(sc_profile_cleaned_path):
        sc_profiles = pd.read_parquet(sc_profile_path)

        # Drop features that contributed to high performance in control
        if drop_control_feat:
            sc_profiles = drop_top_control_feat(sc_profiles, feat_rank_dir, percent_dropping)
                    
        # Drop rows that failed to merge with metadata
        sc_profiles = drop_meta_null(sc_profiles, 'Metadata_Batch')
        
        # Drop features with null values
        sc_profiles = drop_null_features(sc_profiles)
    
        # Save clean profile
        sc_profiles.to_parquet(f"{data_dir}/{batch}{feature_type}_nonull.parquet")
        
    else: 
        sc_profiles = pd.read_parquet(sc_profile_cleaned_path)

    # Get all metadata variable names  
    feat_col = [i for i in sc_profiles.columns if "Metadata_" not in i]
    
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

    controls = sc_profiles[sc_profiles["Metadata_control"].astype('bool')]

    # Drop transfection controls
    controls = controls[controls['Metadata_Symbol']!='516 - TC'].reset_index(drop=True)
    control_group = controls.groupby("Metadata_Sample_Unique").groups

    control_group_runner(
        controls, 
        control_group, 
        result_dir, 
        feat_cols_protein, 
        batch_name=batch,
        protein_prefix='protein')

    experimental_group_runner(
        experiments, 
        gene_group, 
        result_dir, 
        feat_cols_protein, 
        batch_name=batch,
        protein_prefix='protein')
    
    control_group_runner(
        controls, 
        control_group, 
        result_dir, 
        feat_cols_non_protein, 
        batch_name=batch,
        protein_prefix='non_protein')
    
    
    experimental_group_runner(
        experiments, 
        gene_group, 
        result_dir, 
        feat_cols_non_protein, 
        batch_name=batch,
        protein_prefix='non_protein')

main()
            