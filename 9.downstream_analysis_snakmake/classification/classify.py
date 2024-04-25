"""Classification pipeline"""
import numpy as np
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
import xgboost as xgb
import random
from itertools import combinations
from tqdm.contrib.concurrent import thread_map

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("..")
from utils import find_feat_cols, find_meta_cols, remove_nan_infs_columns
import pandas as pd

def drop_top_control_feat(sc_profiles, feat_rank_dir, percent_dropping=0.1):
    """
    This function drops the features with highest weight in previous run of control experiments.
    """
    df_protein_feat_rank = pd.read_csv(f"{feat_rank_dir}/ctrl_protein_feat_rank.csv")
    df_non_protein_feat_rank = pd.read_csv(
        f"{feat_rank_dir}/ctrl_non_protein_feat_rank.csv"
    )

    df_protein_drop = list(
        df_protein_feat_rank["feature"][
            0 : int(df_protein_feat_rank.shape[0] * percent_dropping)
        ]
    )
    df_non_protein_drop = list(
        df_non_protein_feat_rank["feature"][
            0 : int(df_non_protein_feat_rank.shape[0] * percent_dropping)
        ]
    )
    sc_profiles.drop(df_protein_drop + df_non_protein_drop, axis=1, inplace=True)
    print(
        f"Removed {len(df_protein_drop+df_non_protein_drop)} features that dominated control predictions."
    )
    return sc_profiles


def drop_meta_null(sc_profiles, check_col="Metadata_Batch"):
    """
    This function drops the rows that contain null value in metadata (failure in merging).
    """
    row_count = int(sc_profiles.shape[0])
    sc_profiles.drop(np.where(sc_profiles[check_col].isna())[0], axis=0, inplace=True)
    sc_profiles.reset_index(drop=True, inplace=True)
    print(f"Removed {row_count-sc_profiles.shape[0]} rows with NaN metadata values.")
    return sc_profiles


def classifier(
    all_profiles_train,
    all_profiles_test,
    target="Label",
    evaluate=False,
    shuffle=False
):
    """
    This function runs classification.
    """
    feat_col = find_feat_cols(all_profiles_train)
    feat_col.remove(target)
    
    if evaluate:
        x_train, x_val, y_train, y_val = train_test_split(
            all_profiles_train[feat_col],
            all_profiles_train[[target]],
            test_size=0.2,
            random_state=1,
        )
        eval_set = [(x_train, y_train), (x_val, y_val)]

        x_test, y_test = all_profiles_test[feat_col], all_profiles_test[[target]]

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=300,
            tree_method="hist",
            learning_rate=0.05,
            early_stopping_rounds=100,
            device="cuda",
            verbosity=0,
        )

        model.fit(x_train, y_train, eval_set=eval_set, verbose=False)

    else:
        x_train, y_train = all_profiles_train[feat_col], all_profiles_train[[target]]
        x_test, y_test = all_profiles_test[feat_col], all_profiles_test[[target]]

        # x_train_cp, y_train_cp, x_test_cp = cp.array(x_train), cp.array(y_train), cp.array(x_test)
        if shuffle:
            # Create shuffled train labels
            y_train_shuff = y_train.copy()
            y_train_shuff["Label"] = np.random.permutation(y_train.values)

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=150,
            tree_method="hist",
            learning_rate=0.05,
            # device="cuda:7",
        ).fit(x_train, y_train, verbose=False)

    preds = model.predict(x_test)

    # Store feature importance
    feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)

    # Evaluate with metrics
    precision, recall, _ = precision_recall_curve(y_test, preds)
    pr_auc = auc(recall, precision)

    return feat_importances, pr_auc


def get_protein_features(dframe: pd.DataFrame, protein_feat: bool):
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
            if ("GFP" not in i) 
            and ("Brightfield" not in i)
        ]

    dframe = pd.concat([dframe[meta_col], dframe[feat_col]], axis=1)
    return dframe


def stratify_by_plate(df_sampled: pd.DataFrame, plate: str):
    """Stratify dframe by plate"""
    platemap = "_".join(plate.split("T")[0].split("_")[-2:])

    # Train on data from same platemap but other plates
    df_train = df_sampled[
        (df_sampled["Metadata_plate_map_name"] == platemap)
        & (df_sampled["Metadata_Plate"] != plate)
    ].reset_index(drop=True)

    df_test = df_sampled[df_sampled["Metadata_Plate"] == plate].reset_index(drop=True)

    return df_train, df_test


def experimental_runner(
    dframe: pd.DataFrame,
    protein=True,
    group_key_one="Metadata_hgnc_symbol",
    group_key_two="Metadata_gene_allele",
    threshold_key="Metadata_node_type",
):
    """
    Run Reference v.s. Variant experiments
    """
    dframe = get_protein_features(dframe, protein)

    prauc_list = []
    group_list = []
    pair_list = []
    feat_list = []
    plate_name_list = []

    groups = dframe.groupby(group_key_one).groups

    for key in tqdm(groups.keys()):
        dframe_grouped = dframe.loc[groups[key]].reset_index(drop=True)

        # Ensure this gene has both reference and variants
        if dframe_grouped[threshold_key].unique().size != 2:
            continue

        df_group_one = dframe_grouped[
            dframe_grouped["Metadata_node_type"] == "disease_wt"
        ].reset_index(drop=True)
        df_group_one["Label"] = 1

        subgroups = (
            dframe_grouped[dframe_grouped["Metadata_node_type"] == "allele"]
            .groupby(group_key_two)
            .groups
        )

        for subkey in subgroups.keys():
            df_group_two = dframe_grouped.loc[subgroups[subkey]].reset_index(drop=True)
            df_group_two["Label"] = 0

            df_sampled = pd.concat([df_group_one, df_group_two], ignore_index=True)

            plate_list = get_common_plates(df_group_one, df_group_two)

            def classify_by_plate_helper(plate):
                df_train, df_test = stratify_by_plate(df_sampled, plate)
                feat_importances, score = classifier(df_train, df_test)
                return {plate: [feat_importances, score]}

            result = thread_map(classify_by_plate_helper, plate_list)

            for res in result:
                feat_list.append(list(res.values())[0][0])
                group_list.append(key)
                pair_list.append(f"{key}_{subkey}")
                plate_name_list.append(list(res.keys())[0])
                prauc_list.append(list(res.values())[0][1])

    # Store feature importance
    df_feat_one = pd.DataFrame({'Group1': group_list, 'Group2': pair_list})
    df_feat_two = pd.DataFrame(feat_list)
    df_feat = pd.concat([df_feat_one, df_feat_two], axis=1)
    df_feat["Metadata_Protein"] = protein
    df_feat["Metadata_Control"] = False

    df_result = pd.DataFrame(
        {
            'Group1': group_list,
            'Group2': pair_list,
            "Metadata_Plate": plate_name_list,
            "PR_AUC": prauc_list,
        }
    )
    df_result["Metadata_Protein"] = protein
    df_result["Metadata_Control"] = False
    return df_feat, df_result


def get_common_plates(dframe1, dframe2):
    """Helper func: get common plates in two dataframes"""
    plate_list = list(
        set(list(dframe1["Metadata_Plate"].unique()))
        & set(list(dframe2["Metadata_Plate"].unique()))
    )
    return plate_list


def control_group_runner(
    dframe: pd.DataFrame,
    group_key_one="Metadata_hgnc_symbol",
    group_key_two="Metadata_well_position",
    threshold_key="Metadata_well_position",
    protein=True,
):
    """
    Run null control experiments.
    """
    dframe = get_protein_features(dframe, protein)

    prauc_list = []
    group_list = []
    pair_list = []
    feat_list = []
    plate_name_list = []

    groups = dframe.groupby(group_key_one).groups

    for key in tqdm(groups.keys()):
        dframe_grouped = dframe.loc[groups[key]].reset_index(drop=True)

        # Skip controls with no replicates
        if dframe_grouped[threshold_key].unique().size < 2:
            continue

        subgroups = dframe_grouped.groupby(group_key_two).groups

        # Sample 4 out of 6 possible pairwise combinations of well pairs
        sampled_pairs = random.choices(
            list(combinations(list(subgroups.keys()), r=2)), k=4
        )

        for idx1, idx2 in sampled_pairs:
            df_group_one = dframe_grouped.loc[subgroups[idx1]].reset_index(drop=True)
            df_group_one["Label"] = 1
            df_group_two = dframe_grouped.loc[subgroups[idx2]].reset_index(drop=True)
            df_group_two["Label"] = 0
            df_sampled = pd.concat([df_group_one, df_group_two], ignore_index=True)

            plate_list = get_common_plates(df_group_one, df_group_two)

            def classify_by_plate_helper(plate):
                df_train, df_test = stratify_by_plate(df_sampled, plate)
                feat_importances, score = classifier(df_train, df_test)
                return {plate: [feat_importances, score]}

            result = thread_map(classify_by_plate_helper, plate_list)

            for res in result:
                feat_list.append(list(res.values())[0][0])
                group_list.append(key)
                pair_list.append(f"{idx1}_{idx2}")
                plate_name_list.append(list(res.keys())[0])
                prauc_list.append(list(res.values())[0][1])

            # for plate in plate_list:
            #     df_train, df_test = stratify_by_plate(df_sampled, plate)
            #     feat_importances, f1score_macro = classifier(
            #         df_train, df_test
            #     )

            #     feat_list.append(feat_importances)
            #     group_list.append(key)
            #     pair_list.append([idx1,idx2])
            #     plate_list.append(plate)
            #     f1score_list.append(f1score_macro)

    # Store feature importance
    df_feat_one = pd.DataFrame({'Group1': group_list, 'Group2': pair_list})
    df_feat_two = pd.DataFrame(feat_list)
    df_feat = pd.concat([df_feat_one, df_feat_two], axis=1)
    df_feat["Metadata_Protein"] = protein
    df_feat["Metadata_Control"] = True

    df_result = pd.DataFrame(
        {
            'Group1': group_list,
            'Group2': pair_list,
            "Metadata_Plate": plate_name_list,
            "PR_AUC": prauc_list,
        }
    )
    df_result["Metadata_Protein"] = protein
    df_result["Metadata_Control"] = True
    return df_feat, df_result


def control_type_helper(col_annot: str):
    """helper func for annotating column "Metadata_control" """
    if col_annot in ["TC", "NC", "PC", "cPC", "cNC"]:
        return True
    elif col_annot in ["disease_wt", "allele"]:
        return False
    else:
        return None


def add_control_annot(dframe):
    """annotating column "Metadata_control" """
    if "Metadata_control" not in dframe.columns:
        dframe["Metadata_control"] = dframe["Metadata_control_type"].apply(
            lambda x: control_type_helper(x)
        )
    return dframe


def run_classify_workflow(
    input_path: str,
    feat_output_path: str,
    result_output_path: str,
    use_gpu: str | None = "6,7",
):
    """
    Run workflow for single-cell classification
    """
    if use_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu

    dframe = pd.read_parquet(input_path)
    feat_col = find_feat_cols(dframe)

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

    df_exp = dframe[~dframe["Metadata_control"].astype("bool")].reset_index(drop=True)
    df_control = dframe[dframe["Metadata_control"].astype("bool")].reset_index(
        drop=True
    )

    df_control = df_control[df_control["Metadata_control_type"] != "TC"].reset_index(
        drop=True
    )

    df_feat_pro_con, df_result_pro_con = control_group_runner(
        df_control, protein=True
    )
    df_feat_no_pro_con, df_result_no_pro_con = control_group_runner(
        df_control, protein=False
    )
    df_feat_pro_exp, df_result_pro_exp = experimental_runner(
        df_exp, protein=True
    )
    df_feat_no_pro_exp, df_result_no_pro_exp = experimental_runner(
        df_exp, protein=False
    )

    df_feat = pd.concat(
        [df_feat_pro_con, df_feat_no_pro_con, df_feat_pro_exp, df_feat_no_pro_exp],
        ignore_index=True,
    )
    df_result = pd.concat(
        [
            df_result_pro_con,
            df_result_no_pro_con,
            df_result_pro_exp,
            df_result_no_pro_exp,
        ],
        ignore_index=True,
    )

    df_feat.to_csv(feat_output_path, index=False)
    df_result.to_csv(result_output_path, index=False)
