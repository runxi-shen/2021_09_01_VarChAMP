# The B4A3 and B6A3 batches contain 8 different controls, each repeated 48X. 
# There is one WT-VAR pair (ALK) (positive protein loc control), one TC, one positive morphological control (but no matched WT), and four proposed NC.
# The objective is to assess the influence of well position effect using the ALK WT-VAR pair and to evaluate the proposed NC.

import pandas as pd
import polars as pl
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


def classifier(all_profiles_train, all_profiles_test, feat_col, target='Label', stratify_label=None):
    '''
    This function runs classification.
    '''
    X_train, y_train = all_profiles_train[feat_col], all_profiles_train[[target]]
    X_test, y_test = all_profiles_test[feat_col], all_profiles_test[[target]]
    
    model = xgb.XGBClassifier(objective='binary:logistic',
                        n_estimators=150,
                        tree_method='hist',
                        learning_rate = 0.05,
                        device="cuda:7").fit(X_train, y_train, verbose=False)

    preds = model.predict(X_test)

    # Store feature importance
    feat_importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    )
    
    # Evaluate with metrics
    f1score_macro = f1_score(y_test, preds, average="macro")
      
    return feat_importances, f1score_macro



def experimental_group_runner(var_profiles, ref_profiles, var_group, ref_group, data_dir, feat_col, batch_name='', 
    protein_prefix='protein', feature_type='_normalized_feature_selected'):
    '''
    This function runs the reference v.s. variants experiments. 
    '''
    well_list = []
    pair_list = []
    feat_list = []
    f1score_macro_list = []
    result_plate_list = []

    for var_key in tqdm(var_group.keys()):
        
        var_profs = var_profiles.loc[var_group[var_key]]      
        var_profs["Label"] = 1
        
        for ref_key in ref_group.keys():
            ref_profs = ref_profiles.loc[ref_group[ref_key]]
            ref_profs["Label"] = 0
            
            # split data
            var_train_inds = "random sample"
            var_test_inds = "inverse"
            ref_train_inds = "random sample 2"
            ref_test_inds = "inverse 2"
            
            # split into train and test
            all_profiles_train = pd.concat(
                [ref_profs[ref_train_inds]], 
                [var_profs[var_train_inds]], 
                ignore_index=True
                )
            
            all_profiles_test = pd.concat(
                [ref_profs[ref_test_inds]], 
                [var_profs[var_test_inds]], 
                ignore_index=True
                )
            
            # make classifier
            feat_importances, f1score_macro = classifier(all_profiles_train, all_profiles_test, feat_col)
                
            feat_list.append(feat_importances)
            f1score_macro_list.append(f1score_macro)
            gene_list.append(gene_key)
            pair_list.append(var_key)
            result_plate_list.append(plate)


    # Compile results
    df_feat_one = pd.DataFrame({"Gene": well_list, "Variant": pair_list})
    df_feat_two = pd.DataFrame(feat_list)
    df_feat = pd.concat([df_feat_one, df_feat_two], axis=1)

    df_feat.to_csv(f"{data_dir}/{batch_name}_{protein_prefix}_feat_importance{feature_type}.csv", index=False)
    
    result_csv = pd.DataFrame(
    {
        "Gene": gene_list,
        "Variant": pair_list, # remove
        "Plate": result_plate_list, # replace well
        "F1_Score": f1score_macro_list,
        "Cell_Count": ref_profiles.shape[0] # make sure this is the correct number
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
    data_dir = "/dgx1nas1/storage/data/jess/varchamp/sc_data/processed_profiles/B4A3R1"
    feature_type = "_annotated_normalized_featselected"
    batch = "B4A3R1"
    run_name = 'ALK_WT_VAR'
    os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
    
    result_dir = pathlib.Path(f'/dgx1nas1/storage/data/jess/varchamp/sc_data/classification_results/{batch}/{run_name}')
    result_dir.mkdir(exist_ok=True)
    
    sc_profile_path = f"{data_dir}/{batch}{feature_type}.parquet"
    
    # Read in data - only look at ALK wells
    sc_profiles = pl.scan_parquet(sc_profile_path).filter(pl.col("Metadata_SYMBOL") == "ALK")
    
    # Get all metadata variable names  
    feat_col = [i for i in sc_profiles.columns if "Metadata_" not in i]
    
    # columns that we care about are: Metadata_allele and Metadata_Well
    
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

    # Define labels for classification
    variants = sc_profiles.filter(pl.col("Metadata_allele") == 'ALK_R1275Q').collect().to_pandas()
    var_well_group = variants.groupby("Metadata_Well").groups

    references = sc_profiles.filter(pl.col("Metadata_allele") == 'ALK_').collect().to_pandas()
    ref_well_group = references.groupby("Metadata_Well").groups
    
    # Run classification
    control_group_runner(
        references, 
        ref_well_group, 
        result_dir, 
        feat_cols_protein, 
        batch_name=batch,
        protein_prefix='protein')

    experimental_group_runner(
        variants, 
        references,
        var_well_group, 
        ref_well_group,
        result_dir, 
        feat_cols_protein, 
        batch_name=batch,
        protein_prefix='protein')
    
    control_group_runner(
        references, 
        ref_well_group, 
        result_dir, 
        feat_cols_non_protein, 
        batch_name=batch,
        protein_prefix='non_protein')
    
    experimental_group_runner(
        variants, 
        references,
        var_well_group, 
        ref_well_group,
        result_dir, 
        feat_cols_non_protein, 
        batch_name=batch,
        protein_prefix='non_protein')
    
    
if __name__ == '__main__':
    main()