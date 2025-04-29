"""Classification pipeline"""

## Classification pipeline
import os
import re
import io
import sys
import time
import warnings
import multiprocessing
import argparse
from itertools import combinations
from typing import Union
import cupy as cp
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from functools import partial
from sklearn.metrics import roc_auc_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations, product
warnings.filterwarnings("ignore")

## ================================================================================================
sys.path.append("..")
from utils import find_feat_cols, find_meta_cols, remove_nan_infs_columns
CC_THRESHOLD = 20
MAX_WORKERS = 64
## ================================================================================================


def classifier(df_train, df_test, task_id, target="Label", shuffle=False, device="cpu", n_jobs=2):
    """
    This function runs classification.
    """
    feat_col = find_feat_cols(df_train)
    feat_col.remove(target)

    # display(df_train[feat_col])
    x_train, y_train = df_train[feat_col].values, df_train[[target]].values ## cp.array(df_train[feat_col].to_numpy()), cuda version, no need anymore
    x_test, y_test = df_test[feat_col].values, df_test[[target]].values ## cp.array(df_test[feat_col].to_numpy())

    num_pos = df_train[df_train[target] == 1].shape[0]
    num_neg = df_train[df_train[target] == 0].shape[0]

    if (num_pos == 0) or (num_neg == 0):
        print(f"size of pos: {num_pos}, size of neg: {num_neg}")
        feat_importances = pd.Series(np.nan, index=df_train[feat_col].columns)
        return feat_importances, np.nan

    scale_pos_weight = num_neg / num_pos
    if (scale_pos_weight > 100) or (scale_pos_weight < 0.01):
        print(
            f"scale_pos_weight: {scale_pos_weight}, size of pos: {num_pos}, size of neg: {num_neg}"
        )
        feat_importances = pd.Series(np.nan, index=df_train[feat_col].columns)
        return feat_importances, np.nan

    le = LabelEncoder()
    y_train = np.array(le.fit_transform(y_train)) ## cp.array(le.fit_transform(y_train))
    y_test = np.array(le.fit_transform(y_test)) ## cp.array(le.fit_transform(y_test))

    # if shuffle:
    #     # Create shuffled train labels
    #     y_train_shuff = y_train.copy()
    #     y_train_shuff["Label"] = np.random.permutation(y_train.values)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=150,
        tree_method="hist",
        device=device,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        n_jobs=n_jobs
    ).fit(x_train, y_train, verbose=False)

    # get predictions and scores
    pred_score = model.predict_proba(x_test)[:, 1]
    fold_auroc = roc_auc_score(y_test, pred_score) ## y_test.get()

    # Return classifier info
    
    info_0 = df_test[df_test["Label"] == 0].iloc[0]
    info_1 = df_test[df_test["Label"] == 1].iloc[0]
    allele_pair = sorted(pd.concat([df_train, df_test], axis=0)["Metadata_gene_allele"].unique())
    if (len(allele_pair) > 1):
        class_ID = (
            allele_pair[0]
            + "-"
            + allele_pair[1]
            + "_"
            + info_0["Metadata_Plate"]
            + "-"
            + info_0["Metadata_well_position"]
            + "-"
            + info_1["Metadata_well_position"]
        )
    else:
        class_ID = (
            allele_pair[0]
            + "-"
            + allele_pair[0]
            + "_"
            + info_0["Metadata_Plate"]
            + "-"
            + info_0["Metadata_well_position"]
            + "-"
            + info_1["Metadata_well_position"]
        )
    
    # Store feature importance
    feat_importances = pd.Series(
        model.feature_importances_, index=df_train[feat_col].columns
    )

    # Return cell-level predictions
    cellID = df_test.apply(
        lambda row: f"{row['Metadata_Plate']}_{row['Metadata_well_position']}_{row['Metadata_ImageNumber']}_{row['Metadata_ObjectNumber']}",
        axis=1,
    ).to_list()

    pred_df = pd.DataFrame({
        "Classifier_ID": class_ID,
        "TaskID": task_id,
        "CellID": cellID,
        "Label": y_test, ## y_test.get()
        "Prediction": pred_score
    })

    classifier_df = pd.DataFrame({
        "Classifier_ID": [class_ID],
        "TaskID": task_id,
        "Plate": [info_0["Metadata_Plate"]],
        "trainsize_0": [sum(y_train == 0)], ## y_train.get()
        "testsize_0": [sum(y_test == 0)], ## y_test.get()
        "well_0": [info_0["Metadata_well_position"]],
        # "allele_0": [info_0["Metadata_gene_allele"]],
        "trainsize_1": [sum(y_train == 1)], ## y_train.get()
        "testsize_1": [sum(y_test == 1)], ## y_test.get()
        "well_1": [info_1["Metadata_well_position"]],
        # "allele_1": [info_1["Metadata_gene_allele"]],
        "auroc_perm_cv": fold_auroc
    })
    return feat_importances, classifier_df, pred_df


def get_classifier_features(dframe: pd.DataFrame, protein_feat: bool):
    """Helper function to get dframe containing protein or non-protein features"""
    feat_col = find_feat_cols(dframe)
    meta_col = find_meta_cols(dframe)

    if protein_feat:
        feat_col = [
            i
            for i in feat_col
            if ("GFP" in i)
            and ("DNA" not in i)
            and ("AGP" not in i)
            and ("Mito" not in i)
            and ("Brightfield" not in i)
        ]
    else:
        feat_col = [
            i
            for i in feat_col
            if ("GFP" not in i) and ("Brightfield" not in i) ## and ("AGP" not in i)
        ]

    dframe = pd.concat([dframe[meta_col], dframe[feat_col]], axis=1)
    return dframe


def get_common_plates(dframe1, dframe2):
    """Helper func: get common plates in two dataframes"""
    plate_list = list(
        set(list(dframe1["Metadata_Plate"].unique()))
        & set(list(dframe2["Metadata_Plate"].unique()))
    )
    return plate_list


def control_type_helper(col_annot: str):
    """helper func for annotating column "Metadata_control" """
    if col_annot in ["TC", "NC", "PC", "cNC"]:
        return True
    elif col_annot in ["disease_wt", "allele", "cPC"]:
        return False
    else:
        return None


def add_control_annot(dframe):
    """annotating column "Metadata_control" """
    if "Metadata_control" not in dframe.columns:
        dframe["Metadata_control"] = dframe["Metadata_node_type"].apply(
            lambda x: control_type_helper(x)
        )
    return dframe


def drop_low_cc_wells(dframe, cc_thresh, log_file):
    # Drop wells with cell counts lower than the threshold
    dframe["Metadata_Cell_ID"] = dframe.index
    cell_count = (
        dframe.groupby(["Metadata_Plate", "Metadata_Well"])["Metadata_Cell_ID"]
        .count()
        .reset_index(name="Metadata_Cell_Count")
    )
    dframe = dframe.merge(
        cell_count,
        on=["Metadata_Plate", "Metadata_Well"],
    )
    dframe_dropped = (
        dframe[dframe["Metadata_Cell_Count"] < cc_thresh]
    )
    log_file.write(f"Number of Wells dropped due to cell counts < {cc_thresh}: {len((dframe_dropped['Metadata_Plate']+dframe_dropped['Metadata_Well']+dframe_dropped['Metadata_gene_allele']).unique())}\n")
    print(f"Number of Wells dropped due to cell counts < {cc_thresh}: {len((dframe_dropped['Metadata_Plate']+dframe_dropped['Metadata_Well']+dframe_dropped['Metadata_gene_allele']).unique())}")
    dframe_dropped = dframe_dropped.drop_duplicates(subset=["Metadata_Plate", "Metadata_Well"])
    # print(dframe_dropped[["Metadata_Plate","Metadata_Well","Metadata_gene_allele","Metadata_Cell_Count"]])
    if (dframe_dropped.shape[0] > 0):
        for idx in dframe_dropped.index:
            log_file.write(f"{dframe_dropped.loc[idx, 'Metadata_Plate']}, {dframe_dropped.loc[idx, 'Metadata_Well']}:{dframe_dropped.loc[idx, 'Metadata_gene_allele']}\n")
            print(f"{dframe_dropped.loc[idx, 'Metadata_Plate']}, {dframe_dropped.loc[idx, 'Metadata_Well']}:{dframe_dropped.loc[idx, 'Metadata_gene_allele']}\n")
    
    dframe = (
        dframe[dframe["Metadata_Cell_Count"] >= cc_thresh]
        .drop(columns=["Metadata_Cell_Count"])
        .reset_index(drop=True)
    )
    return dframe


#######################################
# 2. LABEL ASSIGNMENT (unchanged)
#######################################
def assign_labels(wells_all, wells_as_ref):
    """
    Given a list of well IDs (e.g. ["A01","A02",...]) and a subset that are reference,
    returns a dict well_id -> "ref" or "var".
    """
    label_map = {}
    for w in wells_all:
        if w in wells_as_ref:
            label_map[w] = "ref"
        else:
            label_map[w] = "var"
    return label_map


#######################################
# 3. BUILD ONE TASK FOR A SINGLE FOLD
#######################################
def build_single_task(df_sampled, label_name, permutation_name, label_map, test_ref, test_var, fold_idx):
    """
    Create a single dictionary describing the train/test setup:
      - 'task_id': f"{permutation_name}_cv_fold{fold_idx}"
      - 'label_map': label_map
      - 'test_ref': test_ref
      - 'test_var': test_var
    The classification function can use these to run the model.
    """
    return {
        "df_sampled": df_sampled,
        "label_name": label_name,
        "task_id": f"{permutation_name}_cv-fold{fold_idx}",
        "label_map": label_map, 
        "test_ref": test_ref,
        "test_var": test_var
    }


#######################################
# 4. GENERATE A BIG LIST OF TASKS
#######################################
def build_perm_cv_tasks_list(df_sampled, label_name, ref_wells, var_wells, num_folds=8, all_permutations=True):
    """
    1) Construct tasks for the *real labeling* (TrueRef).
    2) If all_permutations=True, build tasks for each possible combination
       of wells as ref. (Skipping the real labeling to avoid duplication.)
    3) Return a big list of tasks, each describing 1 fold (train+test).
    """
    wells_all = sorted(set(ref_wells) | set(var_wells))
    real_ref_set = set(ref_wells)

    # Step A. Real labeling tasks
    real_label_map = assign_labels(wells_all, real_ref_set)
    real_perm_name = "TrueRef_" + "-".join(sorted(real_ref_set))

    # Generate all pairs for folds
    # For 4 ref + 4 var = 16 possible (ref_well, var_well) pairs
    # If num_folds < 16, we can sample; else do all
    ref_wells_sorted = sorted([w for w in wells_all if w in real_ref_set])
    var_wells_sorted = [w for w in wells_all if w not in real_ref_set]
    all_pairs = [(r, v) for r in ref_wells_sorted for v in var_wells_sorted]
    if len(all_pairs) > num_folds:
        rng = np.random.default_rng(42)
        chosen_pairs = rng.choice(all_pairs, size=num_folds, replace=False)
    else:
        chosen_pairs = all_pairs

    tasks = []
    # Build tasks for the real labeling
    for fold_idx, (rwell, vwell) in enumerate(chosen_pairs):
        t = build_single_task(df_sampled, label_name, real_perm_name, real_label_map, rwell, vwell, fold_idx)
        tasks.append(t)

    # Step B. If all_permutations=True, build tasks for permutations
    if all_permutations:
        # All combos: pick 4 wells out of 8 => 70 combos
        k_ref = len(ref_wells)
        possible_combos = list(combinations(wells_all, k_ref))
        # skip the real labeling
        possible_combos = [c for c in possible_combos if set(c) != real_ref_set]

        for combo in possible_combos:
            combo_set = set(combo)
            perm_name = "PermRef_" + "-".join(sorted(combo_set))
            label_map_combo = assign_labels(wells_all, combo_set)

            # Now build fold tasks for each pair
            # We'll do the same approach as above
            ref_combo = sorted([w for w in wells_all if w in combo_set])
            var_combo = sorted([w for w in wells_all if w not in combo_set])
            combo_pairs = [(r, v) for r in ref_combo for v in var_combo]

            # If # of pairs > num_folds, we sample; else do all
            if len(combo_pairs) > num_folds:
                rng = np.random.default_rng(42)
                combo_pairs = rng.choice(combo_pairs, size=num_folds, replace=False)

            for fold_idx, (rwell, vwell) in enumerate(combo_pairs):
                t = build_single_task(df_sampled, label_name, perm_name, label_map_combo, rwell, vwell, fold_idx)
                tasks.append(t)

    return tasks
    

#######################################
# 5. RUN A SINGLE TASK
#######################################
def run_single_task(task) -> dict:
    """
    Takes the dictionary describing one permutation+fold, 
    does the train/test classification, returns:
       { task_id: [feat_importances, classifier_df, pred_df] }
    """
    df_sampled = task["df_sampled"]
    label_name = task["label_name"]
    label_map  = task["label_map"]
    test_ref   = task["test_ref"]
    test_var   = task["test_var"]
    task_id    = task["task_id"]

    # Copy df so we can add "Label" safely
    df_local = df_sampled.copy()
    df_local["Label"] = df_local[label_name].map(label_map)
    df_local["Label"] = (df_local["Label"] == "ref").astype(int)

    # Identify train vs. test
    train_ref_wells = [w for w in label_map if label_map[w] == "ref" and w != test_ref]
    train_var_wells = [w for w in label_map if label_map[w] == "var" and w != test_var]
    test_wells      = [test_ref, test_var]

    df_train = df_local[df_local[label_name].isin(train_ref_wells + train_var_wells)].reset_index(drop=True)
    df_test  = df_local[df_local[label_name].isin(test_wells)].reset_index(drop=True)

    # Run classifier
    feat_importances, classifier_df, pred_df = classifier(df_train, df_test, task_id=task_id, target="Label")

    return {f"{task_id}_tr{test_ref}-tv{test_var}": [feat_importances, classifier_df, pred_df]}


#######################################
# 6. TOP-LEVEL RUN FUNCTION
#######################################
def run_all_perm_cv_in_parallel(df_sampled, label_name, ref_wells, var_wells, log_file, num_folds=8, all_permutations=True):
    """
    1) Build one giant list of tasks (each = 1 permutation + 1 fold).
    2) Run them all in parallel with a single thread_map call.
    3) Merge results into a big dictionary.

    Returns dict of
      { f"{permutation_name}_cv_fold{i}": [feat_importances, classifier_df, pred_df], ... }
    """
    start_time = time.time()
    # Build tasks
    tasks = build_perm_cv_tasks_list(df_sampled, label_name, ref_wells, var_wells, num_folds=num_folds, all_permutations=all_permutations)
    # print(tasks[:20])

    # We remove the DataFrame reference from each task if memory is huge
    # => Or keep it if your data is small enough. Possibly store a reference.

    # Run single parallel map
    # Choose an appropriate max_workers to avoid CPU oversubscription
    results_dict = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for task in tasks:
            fut = executor.submit(run_single_task, task)
            futures.append(fut)
        try:
            for fut in as_completed(futures):
                res = fut.result()  # raises if there's an exception in run_single_task
                results_dict.update(res)
        except Exception as e:
            print("Error occurred, shutting down early:", e)
            # This cancels tasks that have not yet started
            executor.shutdown(wait=False, cancel_futures=True)
            # We can return partial results
            return results_dict

    # results_list = thread_map(run_single_task, tasks, max_workers=96, desc="Running tasks", display=False)
    # # Merge all results
    # results_dict = {}
    # for rdict in results_list:
    #     results_dict.update(rdict)
    log_file.write(f"Finished all {len(tasks)} tasks for {tasks[0]['task_id']} in {time.time() - start_time:.2f}s, got {len(results_dict)} results.\n")
    print(f"Finished all {len(tasks)} tasks for {tasks[0]['task_id']} in {time.time() - start_time:.2f}s, got {len(results_dict)} results.")
    return results_dict


#######################################
# 7. HELPER FUNCTIONS FOR PERM RUNNER
#######################################
# Convert letter rows to numbers
def well_to_coordinates(well):
    row_letter, col_number = re.match(r"([A-P])(\d{2})", well).groups()
    row_index = ord(row_letter) - ord('A') + 1  # Convert 'A'->1, 'B'->2, ..., 'P'->16
    col_index = int(col_number)  # Convert string column to integer
    return well, row_index, col_index


# Compute distances from edges and find the most centered well
def compute_distance(row, col):
    return min(row - 1, 16 - row, col - 1, 24 - col)  # Distance from nearest edge


def experimental_perm_runner(
    exp_dframe: pd.DataFrame,
    pq_writer,
    log_file,
    protein,
    group_key_one="Metadata_symbol",
    group_key_two="Metadata_gene_allele"
):
    """
        Run Reference v.s. Variant experiments
    """
    exp_dframe = get_classifier_features(exp_dframe, protein)
    feat_cols = find_feat_cols(exp_dframe)
    feat_cols = [i for i in feat_cols if i != "Label"]
    
    group_list = []
    pair_list = []
    feat_list = []
    info_list = []
    
    log_file.write(f"===============================================================================================================================================================\n")
    log_file.write(f"Logging messages/errors when running real experiments w/ protein {protein}:\n")
    groups = exp_dframe.groupby(group_key_one).groups
    
    var_ref_pred_dict = {}
    label_name = "Metadata_Well_Plate"
    
    for key in tqdm(groups.keys()):
        dframe_grouped = exp_dframe.loc[groups[key]].reset_index(drop=True)
    
        ## get the ref allele, where symbol==allele
        df_group_one = dframe_grouped[
            dframe_grouped[group_key_one] == dframe_grouped[group_key_two]
        ].reset_index(drop=True)
        # df_group_one["Label"] = 1
        
        df_group_one[label_name] = df_group_one["Metadata_well_position"] + '-' + df_group_one["Metadata_Plate"]
        ref_al_wells = df_group_one[label_name].unique()
        
        if len(ref_al_wells) > 4:
            log_file.write(f"{key} was sequenced multiple times:\n{ref_al_wells}\n")
            print(f"{key} was sequenced multiple times:\n{ref_al_wells}")
            # Convert all wells to (name, row, col) tuples
            well_coords = [well_to_coordinates(w) for w in set([ref_well_pl.split('-')[0] for ref_well_pl in ref_al_wells])]
            # Sort wells by max distance from edges (descending)
            most_centered_well = max(well_coords, key=lambda x: compute_distance(x[1], x[2]))[0]
            ref_al_wells = [ref_well_pl for ref_well_pl in ref_al_wells if most_centered_well in ref_well_pl]
            log_file.write(f"Most centered well for {key}:\n{ref_al_wells}\n")
            print(f"Most centered well for {key}:\n{ref_al_wells}\n")
    
        ## get the var alleles, where symbol!=allele
        subgroups = (
            dframe_grouped[dframe_grouped[group_key_one] != dframe_grouped[group_key_two]]
            .groupby(group_key_two)
            .groups
        )
        for subkey in subgroups.keys():
            start_time = time.time()
            df_group_two = dframe_grouped.loc[subgroups[subkey]].reset_index(drop=True)
            df_group_two[label_name] = df_group_two["Metadata_well_position"] + '-' + df_group_two["Metadata_Plate"]
            var_al_wells = df_group_two[label_name].unique()
            df_sampled = pd.concat([df_group_one, df_group_two], ignore_index=True)
            # print(var_al_wells)
    
            if len(ref_al_wells) < 4:
                # ref_al_wells = np.random.choice(ref_al_wells, size=4)
                log_file.write(f"{key}, {subkey} pair DOES NOT enough ref. alleles! Ref. allele wells in parquet: {ref_al_wells}\n")
                continue
            if len(var_al_wells) < 4:
                # var_al_wells = np.random.choice(var_al_wells, size=4)
                log_file.write(f"{key}, {subkey} pair DOES NOT enough var. alleles! Var. allele wells in parquet: {var_al_wells}\n")
                continue
                
            ## try run classifier
            try:
                ctrl_type = list(df_group_two["Metadata_node_type"].unique())[0]
                log_file.write(f"Run inference on WT {key}, VAR {subkey} ({ctrl_type}):\n")
                print(f"Run inference on WT {key}, VAR {subkey} ({ctrl_type}):\n")
                # thread_map returns a list of results, each either a float or None
                results_dict = run_all_perm_cv_in_parallel(df_sampled, label_name, ref_al_wells, var_al_wells, log_file, num_folds=4)
                pred_list = []
                for task_id, dfs in results_dict.items():
                    # print(dfs)
                    ## task_id should be the perm-id_cv-id_var_ref, and dfs are the feat_df, class_df and pred_df per run
                    if len(dfs) == 3:
                        feat_list.append(dfs[0])
                        group_list.append(key)
                        pair_list.append(task_id)
                        info_list.append(dfs[1])
                        pred_list.append(dfs[2])
                    else:
                        print("Result dfs length not 3!")
                        feat_list.append([None] * len(feat_cols))
                        group_list.append(key)
                        pair_list.append(task_id)
                        info_list.append([None] * 10)
    
                cell_preds = pd.concat(pred_list, axis=0)
                cell_preds["Metadata_Protein"] = protein
                cell_preds["Metadata_Control"] = ctrl_type
                table = pa.Table.from_pandas(cell_preds, preserve_index=False)
                pq_writer.write_table(table)
                
                ## store perm df by var vs ref gene pairs if wanted:
                ## =======================================================================================================
                # perm_list = sorted(set([perm.split('_cv-fold')[0] for perm in results_dict.keys()]))
                # ## store the pred_auroc per permutation
                # perm_pred_df = pd.DataFrame()
                # for perm in perm_list:
                #     ## get the CV folds per each permutation
                #     cv_folds_perm = dict([perm_item for perm_item in results_dict.items() if perm in perm_item[0]])
                #     ## get the auroc per each fold of CV
                #     # avg_perm_df = pd.DataFrame()
                #     for cv_fold, classifier_dfs in cv_folds_perm.items():
                #         # print(classifier_dfs[1])
                #         pred_df = classifier_dfs[1]
                #         pred_df["cv_fold"] = cv_fold
                #         pred_df["perm"] = perm
                #         perm_pred_df = pd.concat([perm_pred_df, pred_df])
                #     ## average the auroc across folds of CV for more robust results
                #     # perm_pred_df = pd.concat([perm_pred_df, avg_perm_df], axis=0)
                # var_ref_pred_dict[f"{subkey}-vs-{key}"] = perm_pred_df
                # perm_pred_df.to_csv(f"../../output/permutation_classification_morph_b13/{subkey}_vs_{key}_perm_pred.csv")
                ## =======================================================================================================
            except Exception as e:
                print(e)
                log_file.write(f"{key}, {subkey} error: {e}\n")
        #     break
        # break

    # Store feature importance
    df_feat_one = pd.DataFrame({"Group1": group_list, "Group2": pair_list})
    df_feat_two = pd.DataFrame(feat_list)
    df_feat = pd.concat([df_feat_one, df_feat_two], axis=1)
    df_feat["Metadata_Protein"] = protein
    df_feat["Metadata_Control"] = "allele_cPC_cNC"

    # process classifier info
    df_result = pd.concat(info_list, ignore_index=True)
    df_result["Metadata_Protein"] = protein
    df_result["Metadata_Control"] = "allele_cPC_cNC"
    
    log_file.write(f"Logging messages/errors when running real experiments w/ protein {protein} finished.\n")
    log_file.write(f"===============================================================================================================================================================\n\n")
    return df_feat, df_result


def nc_perm_runner(
    exp_dframe: pd.DataFrame,
    pq_writer,
    log_file,
    protein,
    group_key_one="Metadata_gene_allele",
    group_key_two="Metadata_well_position"
):
    """
        Run Reference v.s. Variant experiments
    """
    exp_dframe = get_classifier_features(exp_dframe, protein)
    feat_cols = find_feat_cols(exp_dframe)
    feat_cols = [i for i in feat_cols if i != "Label"]
    
    group_list = []
    pair_list = []
    feat_list = []
    info_list = []

    log_file.write(f"===============================================================================================================================================================\n")
    log_file.write(f"Logging messages/errors when running NC experiments w/ protein {protein}:\n")
    groups = exp_dframe.groupby(group_key_one).groups
    
    var_ref_pred_dict = {}
    label_name = "Metadata_Well_Plate"
    
    for key in tqdm(groups.keys()):
        dframe_grouped = exp_dframe.loc[groups[key]].reset_index(drop=True)
        ## get the ctrl type
        ctrl_type = list(dframe_grouped["Metadata_node_type"].unique())[0]
        
        unique_wells = dframe_grouped["Metadata_well_position"].unique()
        # Generate all unique pairs
        unique_pairs = list(combinations(unique_wells, 2))
        print(key, "Unique well pairs:", unique_pairs)
        
        for ref_well, var_well in unique_pairs:
            # print(ref_well, var_well)
            start_time = time.time()

            for plate in ["P1", "P2"]:
                ## get the allele for NC
                df_group_one = dframe_grouped[
                    (dframe_grouped[group_key_two]==ref_well)&(dframe_grouped["Metadata_Plate"].str.contains(plate))
                ].reset_index(drop=True)
            
                df_group_one[label_name] = df_group_one["Metadata_well_position"] + '-' + df_group_one["Metadata_Plate"]
                ref_al_wells = df_group_one[label_name].unique()
        
                df_group_two = dframe_grouped.loc[
                    (dframe_grouped[group_key_two]==var_well)&(dframe_grouped["Metadata_Plate"].str.contains(plate))
                ].reset_index(drop=True)
                df_group_two[label_name] = df_group_two["Metadata_well_position"] + '-' + df_group_two["Metadata_Plate"]
                var_al_wells = df_group_two[label_name].unique()
    
                # print(ref_al_wells, var_al_wells)
                df_sampled = pd.concat([df_group_one, df_group_two], ignore_index=True)
                # print(var_al_wells)
        
                if len(ref_al_wells) < 4:
                    # ref_al_wells = np.random.choice(ref_al_wells, size=4)
                    log_file.write(f"{key} pair DOES NOT enough ref. alleles! Ref. allele wells in parquet: {ref_al_wells}\n")
                    continue
                if len(var_al_wells) < 4:
                    # var_al_wells = np.random.choice(var_al_wells, size=4)
                    log_file.write(f"{key} pair DOES NOT enough var. alleles! Var. allele wells in parquet: {var_al_wells}\n")
                    continue
                    
                ## try run classifier
                try:
                    log_file.write(f"Run inference on {key} ({ctrl_type}):\n")
                    print(f"Run inference on {key} ({ctrl_type}):\n")
                    # thread_map returns a list of results, each either a float or None
                    results_dict = run_all_perm_cv_in_parallel(df_sampled, label_name, ref_al_wells, var_al_wells, log_file, num_folds=4)
                    pred_list = []
                    for task_id, dfs in results_dict.items():
                        # print(dfs)
                        ## task_id should be the perm-id_cv-id_var_ref, and dfs are the feat_df, class_df and pred_df per run
                        if len(dfs) == 3:
                            feat_list.append(dfs[0])
                            group_list.append(key)
                            pair_list.append(task_id)
                            info_list.append(dfs[1])
                            pred_list.append(dfs[2])
                        else:
                            print("Result dfs length not 3!")
                            feat_list.append([None] * len(feat_cols))
                            group_list.append(key)
                            pair_list.append(task_id)
                            info_list.append([None] * 10)
        
                    cell_preds = pd.concat(pred_list, axis=0)
                    cell_preds["Metadata_Protein"] = protein
                    cell_preds["Metadata_Control"] = ctrl_type
                    table = pa.Table.from_pandas(cell_preds, preserve_index=False)
                    pq_writer.write_table(table)
                except Exception as e:
                    print(e)
                    log_file.write(f"{key}, {key} error: {e}")
        #         break
        #     break
        # break

    # Store feature importance
    df_feat_one = pd.DataFrame({"Group1": group_list, "Group2": pair_list})
    df_feat_two = pd.DataFrame(feat_list)
    df_feat = pd.concat([df_feat_one, df_feat_two], axis=1)
    df_feat["Metadata_Protein"] = protein
    df_feat["Metadata_Control"] = ctrl_type

    # process classifier info
    df_result = pd.concat(info_list, ignore_index=True)
    df_result["Metadata_Control"] = ctrl_type

    log_file.write(f"Logging messages/errors when running NC experiments w/ protein {protein} finished.\n")
    log_file.write(f"===============================================================================================================================================================\n\n")
    return df_feat, df_result


def pc_perm_runner(
    exp_dframe: pd.DataFrame,
    pq_writer,
    log_file,
    protein,
    group_key_one="Metadata_symbol",
    group_key_two="Metadata_gene_allele"
):
    """
        Run Reference v.s. Variant experiments
    """
    exp_dframe = get_classifier_features(exp_dframe, protein)
    feat_cols = find_feat_cols(exp_dframe)
    feat_cols = [i for i in feat_cols if i != "Label"]
    
    group_list = []
    pair_list = []
    feat_list = []
    info_list = []

    log_file.write(f"===============================================================================================================================================================\n")
    log_file.write(f"Logging messages/errors when running PC experiments w/ protein {protein}:\n")
    groups = exp_dframe.groupby(group_key_one).groups
    
    var_ref_pred_dict = {}
    label_name = "Metadata_Well_Plate"
    
    for key in tqdm(groups.keys()):
        dframe_grouped = exp_dframe.loc[groups[key]].reset_index(drop=True)
        ## get the ctrl type
        ctrl_type = list(dframe_grouped["Metadata_node_type"].unique())[0]
        
        dframe_ref = dframe_grouped[dframe_grouped[group_key_one] == dframe_grouped[group_key_two]]
        dframe_var = dframe_grouped[dframe_grouped[group_key_one] != dframe_grouped[group_key_two]]

        ref_wells = dframe_ref["Metadata_well_position"].unique()
        var_wells = dframe_var["Metadata_well_position"].unique()

        subkey = list(dframe_var["Metadata_gene_allele"].unique())[0]
        print(f"{key} Ref wells:", ref_wells, f"{subkey} Var wells:", var_wells)
        log_file.write(f"{key} Ref wells: {ref_wells};\n")
        log_file.write(f"{subkey} Var wells: {var_wells}\n")
        # Generate all unique pairs
        unique_pairs = list(product(ref_wells, var_wells))
        
        for ref_well, var_well in unique_pairs:
            print("Selected ref:", ref_well, "| Selected var:", var_well)
            start_time = time.time()

            for plate in ["P1", "P2"]:
                ## get the allele for PC
                df_group_one = dframe_grouped[
                    (dframe_grouped["Metadata_well_position"]==ref_well)&(dframe_grouped["Metadata_Plate"].str.contains(plate))
                ].reset_index(drop=True)
            
                df_group_one[label_name] = df_group_one["Metadata_well_position"] + '-' + df_group_one["Metadata_Plate"]
                ref_al_wells = df_group_one[label_name].unique()
        
                df_group_two = dframe_grouped.loc[
                    (dframe_grouped["Metadata_well_position"]==var_well)&(dframe_grouped["Metadata_Plate"].str.contains(plate))
                ].reset_index(drop=True)
                df_group_two[label_name] = df_group_two["Metadata_well_position"] + '-' + df_group_two["Metadata_Plate"]
                var_al_wells = df_group_two[label_name].unique()
                # print(ref_al_wells, var_al_wells)
                
                df_sampled = pd.concat([df_group_one, df_group_two], ignore_index=True)    
                if len(ref_al_wells) < 4:
                    # ref_al_wells = np.random.choice(ref_al_wells, size=4)
                    log_file.write(f"{key}, {subkey} pair DOES NOT enough ref. alleles! Ref. allele wells in parquet: {ref_al_wells}\n")
                    continue
                if len(var_al_wells) < 4:
                    # var_al_wells = np.random.choice(var_al_wells, size=4)
                    log_file.write(f"{key}, {subkey} pair DOES NOT enough var. alleles! Var. allele wells in parquet: {var_al_wells}\n")
                    continue
                    
                ## try run classifier
                try:
                    # log_file.write(f"Run inference on WT {key} REF Wells: {ref_al_wells}), VAR {subkey} Wells: {var_al_wells}...\n")
                    # print(f"Run inference on WT {key} REF Wells: {ref_al_wells}), VAR {subkey} Wells: {var_al_wells}...")
                    # thread_map returns a list of results, each either a float or None
                    results_dict = run_all_perm_cv_in_parallel(df_sampled, label_name, ref_al_wells, var_al_wells, log_file, num_folds=4)
                    pred_list = []
                    for task_id, dfs in results_dict.items():
                        # print(dfs)
                        ## task_id should be the perm-id_cv-id_var_ref, and dfs are the feat_df, class_df and pred_df per run
                        if len(dfs) == 3:
                            feat_list.append(dfs[0])
                            group_list.append(key)
                            pair_list.append(task_id)
                            info_list.append(dfs[1])
                            pred_list.append(dfs[2])
                        else:
                            print("Result dfs length not 3!")
                            feat_list.append([None] * len(feat_cols))
                            group_list.append(key)
                            pair_list.append(task_id)
                            info_list.append([None] * 10)
        
                    cell_preds = pd.concat(pred_list, axis=0)
                    cell_preds["Metadata_Protein"] = protein
                    cell_preds["Metadata_Control"] = ctrl_type
                    table = pa.Table.from_pandas(cell_preds, preserve_index=False)
                    pq_writer.write_table(table)
                    print(f"Run inference on Gene {key} ({ctrl_type}) with REF: {ref_well}, VAR: {var_well} takes {start_time - time.time()}s.")
                except Exception as e:
                    print(e)
                    log_file.write(f"{key}, {key} error: {e}")
        #         break
        #     break
        # break

    # Store feature importance
    df_feat_one = pd.DataFrame({"Group1": group_list, "Group2": pair_list})
    df_feat_two = pd.DataFrame(feat_list)
    df_feat = pd.concat([df_feat_one, df_feat_two], axis=1)
    df_feat["Metadata_Protein"] = protein
    df_feat["Metadata_Control"] = ctrl_type

    # process classifier info
    df_result = pd.concat(info_list, ignore_index=True)
    df_result["Metadata_Control"] = ctrl_type

    log_file.write(f"Logging messages/errors when running PC experiments w/ protein {protein} finished.\n")
    log_file.write(f"===============================================================================================================================================================\n\n")
    return df_feat, df_result


def run_perm_workflow(
    input_path: str,
    feat_output_path: str,
    info_output_path: str,
    preds_output_path: str,
    logfile_path: str,
    cc_threshold: int
):
    """
        Run workflow for single-cell classification
    """
    ## INPUT ARGUMENTS FOR CLASSIFICATION
    # input_path = "../../../6.downstream_analysis_snakemake/outputs/batch_profiles/2025_01_27_Batch_13/profiles_tcdropped_filtered_var_mad_outlier_featselect_correct_meta.parquet"
    # output_path = "../../output/classify_reimplement/testing"
        
    # Initialize parquet for cell-level predictions
    if os.path.exists(preds_output_path):
        os.remove(preds_output_path)
        
    schema = pa.schema([
        ("Classifier_ID", pa.string()),
        ("TaskID", pa.string()),
        ("CellID", pa.string()),
        ("Label", pa.int64()),
        ("Prediction", pa.float32()),
        ("Metadata_Protein", pa.bool_()),
        ("Metadata_Control", pa.string()),
    ])
    writer = pq.ParquetWriter(preds_output_path, schema, compression="gzip")
    
    # Add CellID column
    dframe = (
        pl.scan_parquet(input_path)
        .with_columns(
            ## Reformat plate_map
            pl.col("Metadata_plate_map_name")
            .str.split("_")  # Split the string by '_'
            .list.get(3)     # Get the third element (index 2)
            .alias("Metadata_plate_map_name"),
            pl.concat_str(
                [
                    "Metadata_Plate",
                    "Metadata_well_position",
                    "Metadata_ImageNumber",
                    "Metadata_ObjectNumber",
                ],
                separator="_",
            ).alias("Metadata_CellID")
        )
        .collect()
        .to_pandas()
    )
    feat_col = find_feat_cols(dframe)
    # print(f"Number of features: {len(feat_col)}\n", feat_col[:10])
    
    try:
        assert (
            ~np.isnan(dframe[feat_col]).any().any()
        ), "Dataframe contains no NaN features."
        assert (
            np.isfinite(dframe[feat_col]).all().all()
        ), "Dataframe contains finite feature values."
    except AssertionError:
        dframe = remove_nan_infs_columns(dframe)
    
    # Filter rows with NaN Metadata
    dframe = dframe[~dframe["Metadata_well_position"].isna()]
    dframe = add_control_annot(dframe)
    dframe = dframe[~dframe["Metadata_control"].isna()]
    
    # Split data into controls and alleles
    df_exp = dframe[~dframe["Metadata_control"].astype("bool")].reset_index(drop=True)
    df_control = dframe[dframe["Metadata_control"].astype("bool")].reset_index(
        drop=True
    )
    # Remove any remaining TC from analysis
    df_control = df_control[df_control["Metadata_node_type"] != "TC"].reset_index(
        drop=True
    )

    # Open log file for writing
    with open(logfile_path, "w") as log_file:
        # log_file.write(f"===============================================================================================================================================================\n")
        # log_file.write("Dropping low cell count wells in ref. vs variant alleles:\n")
        # print("Dropping low cell count wells in ref. vs variant alleles:\n")
        # df_exp = drop_low_cc_wells(df_exp, cc_threshold, log_file)
        # log_file.write(f"===============================================================================================================================================================\n\n")
        # print(f"===============================================================================================================================================================\n\n")

        log_file.write(f"===============================================================================================================================================================\n")
        log_file.write("Dropping low cell count wells in control wells:\n")
        print("Dropping low cell count wells in control wells:\n")
        df_control = drop_low_cc_wells(df_control, cc_threshold, log_file)
        log_file.write(f"===============================================================================================================================================================\n\n")
        print(f"===============================================================================================================================================================\n\n")

        ## Protein feature analysis
        # df_feat_pro, df_result_pro = experimental_perm_runner(
        #     df_exp, pq_writer=writer, log_file=log_file, protein=True
        # )
        ## Non-protein feature analysis
        df_feat_no_pro, df_result_no_pro = experimental_perm_runner(
            df_exp, pq_writer=writer, log_file=log_file, protein=False
        )

        ## Protein feature analysis
        # df_feat_nc_pro, df_result_nc_pro = nc_perm_runner(
        #     df_control[df_control["Metadata_node_type"] == "NC"], pq_writer=writer, log_file=log_file, protein=True
        # )
        # df_feat_pc_pro, df_result_pc_pro = pc_perm_runner(
        #     df_control[df_control["Metadata_node_type"] == "PC"], pq_writer=writer, log_file=log_file, protein=True
        # )

        ## Non-protein feature analysis
        df_feat_nc_no_pro, df_result_nc_no_pro = nc_perm_runner(
            df_control[df_control["Metadata_node_type"] == "NC"], pq_writer=writer, log_file=log_file, protein=False
        )
        df_feat_pc_no_pro, df_result_pc_no_pro = pc_perm_runner(
            df_control[df_control["Metadata_node_type"] == "PC"], pq_writer=writer, log_file=log_file, protein=False
        )

    # Concatenate results for both protein and non-protein
    df_feat = pd.concat(
        [
            # df_feat_pro,
            df_feat_no_pro,
            # df_feat_nc_pro,
            df_feat_nc_no_pro,
            # df_feat_pc_pro,
            df_feat_pc_no_pro
        ],
        ignore_index=True,
    )
    df_result = pd.concat(
        [
            # df_result_pro,
            df_result_no_pro,
            # df_result_nc_pro,
            df_result_nc_no_pro,
            # df_result_pc_pro,
            df_result_pc_no_pro
        ],
        ignore_index=True,
    )
    df_result = df_result.drop_duplicates()

    # Write out feature importance and classifier info
    df_feat.to_csv(feat_output_path, index=False)
    df_result.to_csv(info_output_path, index=False)


def main():
    """
        Run workflow for single-cell classification
    """
    ## INPUT ARGUMENTS FOR CLASSIFICATION
    parser = argparse.ArgumentParser(description="Taking in input file and output dir.")
    parser.add_argument('--input_path', type=str, required=True, help='Input parquet file')
    parser.add_argument('--output_path', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    # input_path = "/home/shenrunx/igvf/varchamp/2021_09_01_VarChAMP/6.downstream_analysis_snakemake/outputs/batch_profiles/2025_01_28_Batch_14/profiles_tcdropped_filtered_var_mad_outlier_featselect_correct_meta.parquet"
    # output_path = "../../output/classify_reimplement/2025_01_28_Batch_14/profiles_tcdropped_filtered_var_mad_outlier_featselect_correct_meta_noAGP"
    
    if not os.path.exists(args.output_path):
        print(f"{args.output_path} doesn't exist. Creating it...")
        os.makedirs(args.output_path)
    
    preds_output_path = f"{args.output_path}/predictions.parquet"
    feat_output_path = f"{args.output_path}/feat_importance.csv"
    info_output_path = f"{args.output_path}/classifier_info.csv"
    logfile_path = f"{args.output_path}/logfile.log"

    # Initialize parquet for cell-level predictions
    if os.path.exists(preds_output_path):
        os.remove(preds_output_path)

    run_perm_workflow(args.input_path, 
                     feat_output_path=feat_output_path, 
                     info_output_path=info_output_path,
                     preds_output_path=preds_output_path,
                     logfile_path=logfile_path,
                     cc_threshold=CC_THRESHOLD)


if __name__ == '__main__':
    main()
    