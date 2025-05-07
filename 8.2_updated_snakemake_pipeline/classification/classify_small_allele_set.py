## Classification pipeline
## Author: Runxi Shen, Jess Ewald
## ==============================

import os
import io
import sys
import argparse
import warnings
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
warnings.filterwarnings("ignore")
sys.path.append("..")
from utils import find_feat_cols, find_meta_cols, remove_nan_infs_columns


############################################################
# 1. UNCHANGED classifier and util functions defined by Jess
############################################################
def classifier(df_train, df_test, log_file, target="Label", shuffle=False):
    """
    This function runs classification.
    """
    feat_col = find_feat_cols(df_train)
    feat_col.remove(target)

    x_train, y_train = cp.array(df_train[feat_col].to_numpy()), df_train[[target]]
    x_test, y_test = cp.array(df_test[feat_col].to_numpy()), df_test[[target]]

    num_pos = df_train[df_train[target] == 1].shape[0]
    num_neg = df_train[df_train[target] == 0].shape[0]

    if (num_pos == 0) or (num_neg == 0):
        log_file.write(f"Missing positive/negative labels in {df_train['Metadata_Plate'].unique()}, {df_train['Metadata_symbol'].unique()} wells: {df_train['Metadata_well_position'].unique()}\n")
        log_file.write(f"Size of pos: {num_pos}, Size of neg: {num_neg}\n")

        print(f"size of pos: {num_pos}, size of neg: {num_neg}")
        feat_importances = pd.Series(np.nan, index=df_train[feat_col].columns)
        return feat_importances, np.nan

    scale_pos_weight = num_neg / num_pos

    if (scale_pos_weight > 100) or (scale_pos_weight < 0.01):
        log_file.write(f"Extreme class imbalance in {df_train['Metadata_Plate'].unique()}, {df_train['Metadata_symbol'].unique()} wells: {df_train['Metadata_well_position'].unique()}\n")
        log_file.write(f"Scale_pos_weight: {scale_pos_weight}, Size of pos: {num_pos}, Size of neg: {num_neg}\n")
        print(
            f"scale_pos_weight: {scale_pos_weight}, size of pos: {num_pos}, size of neg: {num_neg}"
        )
        feat_importances = pd.Series(np.nan, index=df_train[feat_col].columns)
        return feat_importances, np.nan

    le = LabelEncoder()
    y_train = cp.array(le.fit_transform(y_train))
    y_test = cp.array(le.fit_transform(y_test))

    if shuffle:
        # Create shuffled train labels
        y_train_shuff = y_train.copy()
        y_train_shuff["Label"] = np.random.permutation(y_train.values)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=150,
        tree_method="hist",
        device="cuda",
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
    ).fit(x_train, y_train, verbose=False)

    # get predictions and scores
    pred_score = model.predict_proba(x_test)[:, 1]

    # Return classifier info
    info_0 = df_test[df_test["Label"] == 0].iloc[0]
    info_1 = df_test[df_test["Label"] == 1].iloc[0]
    class_ID = (
        info_0["Metadata_Plate"]
        + "_"
        + info_0["Metadata_well_position"]
        + "_"
        + info_1["Metadata_well_position"]
    )
    classifier_df = pd.DataFrame({
        "Classifier_ID": [class_ID],
        "Plate": [info_0["Metadata_Plate"]],
        "trainsize_0": [sum(y_train.get() == 0)],
        "testsize_0": [sum(y_test.get() == 0)],
        "well_0": [info_0["Metadata_well_position"]],
        "allele_0": [info_0["Metadata_gene_allele"]],
        "trainsize_1": [sum(y_train.get() == 1)],
        "testsize_1": [sum(y_test.get() == 1)],
        "well_1": [info_1["Metadata_well_position"]],
        "allele_1": [info_1["Metadata_gene_allele"]],
    })

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
        "CellID": cellID,
        "Label": y_test.get(),
        "Prediction": pred_score,
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
            if ("GFP" not in i) and ("Brightfield" not in i)
        ]

    dframe = pd.concat([dframe[meta_col], dframe[feat_col]], axis=1)
    return dframe


# def get_common_plates(dframe1, dframe2):
#     """Helper func: get common plates in two dataframes"""
#     plate_list = list(
#         set(list(dframe1["Metadata_Plate"].unique()))
#         & set(list(dframe2["Metadata_Plate"].unique()))
#     )
#     return plate_list


# def control_type_helper(col_annot: str):
#     """helper func for annotating column "Metadata_control" """
#     ## Only TC, NC, PC are used for constructing the null distribution because of multiple duplicates 
#     if col_annot in ["TC", "NC", "PC"]:
#         return True
#     ## else labeled as not controls
#     elif col_annot in ["disease_wt", "allele", "cPC", "cNC"]:
#         return False
#     else:
#         return None


# def add_control_annot(dframe):
#     """Annotating column "Metadata_control" """
#     if "Metadata_control" not in dframe.columns:
#         dframe["Metadata_control"] = dframe["Metadata_node_type"].apply(
#             lambda x: control_type_helper(x)
#         )
#     return dframe


# def drop_low_cc_wells(dframe, cc_thresh):
#     # Drop wells with cell counts lower than the threshold
#     dframe["Metadata_Cell_ID"] = dframe.index
#     cell_count = (
#         dframe.groupby(["Metadata_Plate", "Metadata_Well"])["Metadata_Cell_ID"]
#         .count()
#         .reset_index(name="Metadata_Cell_Count")
#     )
#     dframe = dframe.merge(
#         cell_count,
#         on=["Metadata_Plate", "Metadata_Well"],
#     )
#     dframe_dropped = (
#         dframe[dframe["Metadata_Cell_Count"] < cc_thresh]
#     )
#     print(f"Wells dropped due to cell counts < {cc_thresh}: {len(dframe_dropped['Metadata_Well'].unique())}")
#     dframe_dropped = dframe_dropped.drop_duplicates(subset="Metadata_Well")
#     # print(dframe_dropped[["Metadata_Plate","Metadata_Well","Metadata_gene_allele","Metadata_Cell_Count"]])
#     well_gene_pair = dict(zip(dframe_dropped["Metadata_Well"].to_list(), 
#                               dframe_dropped["Metadata_gene_allele"].to_list()))
#     if (well_gene_pair):
#         for well, well_gene in well_gene_pair.items():
#             print(f"{well}:{well_gene}")
    
#     dframe = (
#         dframe[dframe["Metadata_Cell_Count"] >= cc_thresh]
#         .drop(columns=["Metadata_Cell_Count"])
#         .reset_index(drop=True)
#     )
#     return dframe


#######################################
# 2. RUN CLASSIFIERS ON CTRL ALLELES
# Resampling ctrl wells and run the 
# classifiers on them for the null dist.
#######################################
def stratify_by_well_pair_ctrl(dframe_grouped_two: pd.DataFrame, well_pair_trn: tuple):
    """Stratify dframe by ctrl well pairs: one pair for training and one pair for testing"""
    sub_sub_groups = dframe_grouped_two.groupby("Metadata_well_position").groups
    assert len(sub_sub_groups.keys()) == 4, f"Number of wells per plate is not 4: {sub_sub_groups.keys()}"

    ## Train on data from well_pair_trn
    df_group_one = dframe_grouped_two.loc[sub_sub_groups[well_pair_trn[0]]].reset_index(
        drop=True
    )
    df_group_one["Label"] = 1
    df_group_two = dframe_grouped_two.loc[sub_sub_groups[well_pair_trn[1]]].reset_index(
        drop=True
    )
    df_group_two["Label"] = 0
    df_sampled = pd.concat([df_group_one, df_group_two], ignore_index=True)
    df_train = df_sampled.reset_index(drop=True)

    ## Test on data from well_pair_test
    well_pair_test = tuple(key for key in sub_sub_groups.keys() if key not in well_pair_trn)
    df_group_3 = dframe_grouped_two.loc[sub_sub_groups[well_pair_test[0]]].reset_index(
        drop=True
    )
    df_group_3["Label"] = 1
    df_group_4 = dframe_grouped_two.loc[sub_sub_groups[well_pair_test[1]]].reset_index(
        drop=True
    )
    df_group_4["Label"] = 0
    df_sampled_test = pd.concat([df_group_3, df_group_4], ignore_index=True)
    df_test = df_sampled_test.reset_index(drop=True)
    return df_train, df_test


def classify_by_well_pair_helper(df_sampled: pd.DataFrame, well_pair: tuple):
    """Helper func to run classifiers in parallel"""
    df_train, df_test = stratify_by_well_pair_ctrl(df_sampled, well_pair)
    feat_importances, classifier_info, predictions = classifier(
        df_train, df_test
    )
    return {f"trn_{well_pair[0]}_{well_pair[1]}": [feat_importances, classifier_info, predictions]}


def control_group_runner_fewer_rep(
    ctrl_dframe: pd.DataFrame,
    pq_writer,
    err_logger,
    group_key_one="Metadata_gene_allele",
    group_key_two="Metadata_plate_map_name",
    group_key_three="Metadata_well_position",
    threshold_key="Metadata_well_position",
    protein=True,
    well_count_min=None
):
    """
    Run classifiers on control alleles.

    # df_feat_pro_con, df_result_pro_con = control_group_runner_fewer_rep(df_control, pq_writer=writer, protein=True/False)
    """
    ctrl_dframe = get_classifier_features(ctrl_dframe, protein)
    feat_cols = find_feat_cols(ctrl_dframe)
    feat_cols = [i for i in feat_cols if i != "Label"]

    group_list = []
    pair_list = []
    feat_list = []
    info_list = []

    err_logger.write(f"Logging errors when running control experiments w/ protein {protein}:\n")
    ## first we group the cells from the same Metadata_gene_allele
    groups = ctrl_dframe.groupby(group_key_one).groups
    for key in tqdm(groups.keys()):
        ## groupby alleles
        dframe_grouped = ctrl_dframe.loc[groups[key]].reset_index(drop=True)
        # Skip controls with no replicates
        if dframe_grouped[threshold_key].unique().size < 2:
            continue
        ## group by platemap
        subgroups = dframe_grouped.groupby(group_key_two).groups
        for key_two in subgroups.keys():
            ## for each platemap
            dframe_grouped_two = dframe_grouped.loc[subgroups[key_two]].reset_index(
                drop=True
            )
            ## If a well is not present on all four plates, drop well
            ## ONLY used when we have enough TECHNICAL-REPLICATE plates!!!
            if well_count_min is not None:
                well_count = dframe_grouped_two.groupby(["Metadata_Well"])[
                    "Metadata_Plate"
                ].nunique()
                well_to_drop = well_count[well_count < well_count_min].index
                dframe_grouped_two = dframe_grouped_two[
                    ~dframe_grouped_two["Metadata_Well"].isin(well_to_drop)
                ].reset_index(drop=True)

            ## group by well
            sub_sub_groups = dframe_grouped_two.groupby(group_key_three).groups
            sampled_pairs = list(combinations(list(sub_sub_groups.keys()), r=2))
            ## juxtapose each pair of wells against each other                
            try:
                ## Bind df_sampled to the helper function
                classify_by_well_pair_bound = partial(classify_by_well_pair_helper, dframe_grouped_two)
                result = thread_map(classify_by_well_pair_bound, sampled_pairs)
                pred_list = []
                for res in result:
                    if len(list(res.values())[0]) == 3:
                        feat_list.append(list(res.values())[0][0])
                        group_list.append(key)
                        pair_list.append(list(res.keys())[0])
                        info_list.append(list(res.values())[0][1])
                        pred_list.append(list(res.values())[0][2])
                    else:
                        print("res length does not equal three!")
                        feat_list.append([None] * len(feat_cols))
                        group_list.append(key)
                        pair_list.append(list(res.keys())[0])
                        info_list.append([None] * 10)

                cell_preds = pd.concat(pred_list, axis=0)
                cell_preds["Metadata_Protein"] = protein
                cell_preds["Metadata_Control"] = True
                table = pa.Table.from_pandas(cell_preds, preserve_index=False)
                pq_writer.write_table(table)
            except Exception as e:
                print(e)
                err_logger.write(f"{key}, {key_two} error: {e}, wells per ctrl: {sub_sub_groups}\n")

    # Store feature importance
    df_feat_one = pd.DataFrame({"Group1": group_list, "Group2": pair_list})
    df_feat_two = pd.DataFrame(feat_list)
    df_feat = pd.concat([df_feat_one, df_feat_two], axis=1)
    df_feat["Metadata_Protein"] = protein
    df_feat["Metadata_Control"] = True

    # process classifier info
    df_result = pd.concat(info_list, ignore_index=True)
    df_result["Metadata_Control"] = True

    err_logger.write(f"Logging errors when running control experiments w/ protein {protein} finished.\n")
    err_logger.write(f"==============================================================================\n\n")
    return df_feat, df_result
    

#######################################
# 3. RUN CLASSIFIERS ON VAR-REF ALLELES
# Construct 4-fold CV on var-vs-ref wells 
# and run the classifiers on them.
#######################################
def stratify_by_well_pair_exp(df_sampled: pd.DataFrame, well_pair_list: list):
    """
        Stratify dframe by plate
        df_sampled: the data frame containing both ref. and var. alleles, each tested in 4 wells
        well_pair: a list of well pairs containing a ref. and a var. allele, with 1st pair for test and the rest pairs for training
    """
    ## Training on the rest three wells out of four
    df_train = df_sampled[
        (df_sampled["Metadata_well_position"].isin([well for pair in well_pair_list[1:] for well in pair]))
    ].reset_index(drop=True)
    ## Testing on the well_pair
    df_test = df_sampled[
        df_sampled["Metadata_well_position"].isin(well_pair_list[0])
    ].reset_index(drop=True)
    return df_train, df_test


def classify_by_well_pair_exp_helper(df_sampled: pd.DataFrame, well_pair_list: list):
    """Helper func to run classifiers in parallel for var-ref alleles"""
    df_train, df_test = stratify_by_well_pair_exp(df_sampled, well_pair_list)
    feat_importances, classifier_info, predictions = classifier(
        df_train, df_test
    )
    well_pair = well_pair_list[0]
    return {f"test_{well_pair[0]}_{well_pair[1]}": [feat_importances, classifier_info, predictions]}


def experimental_runner_plate_rep(
    exp_dframe: pd.DataFrame,
    pq_writer,
    err_logger,
    protein=True,
    group_key_one="Metadata_symbol",
    group_key_two="Metadata_gene_allele",
    threshold_key="Metadata_node_type",
):
    """
    Run Reference v.s. Variant experiments run on the same plate without tech. dups

    # df_feat_pro_exp, df_result_pro_exp = experimental_runner_plate_rep(df_exp, pq_writer=writer, protein=True)
    """
    exp_dframe = get_classifier_features(exp_dframe, protein)
    feat_cols = find_feat_cols(exp_dframe)
    feat_cols = [i for i in feat_cols if i != "Label"]

    group_list = []
    pair_list = []
    feat_list = []
    info_list = []

    err_logger.write(f"Logging errors when running real experiments w/ protein {protein}:\n")
    groups = exp_dframe.groupby(group_key_one).groups
    for key in tqdm(groups.keys()):
        dframe_grouped = exp_dframe.loc[groups[key]].reset_index(drop=True)

        # Ensure this gene has both reference and variants
        if dframe_grouped[threshold_key].unique().size < 2:
            continue

        df_group_one = dframe_grouped[
            dframe_grouped["Metadata_node_type"] == "disease_wt"
        ].reset_index(drop=True)
        df_group_one["Label"] = 1
        ref_al_wells = df_group_one["Metadata_well_position"].unique()
        
        subgroups = (
            dframe_grouped[dframe_grouped["Metadata_node_type"] == "allele"]
            .groupby(group_key_two)
            .groups
        )

        for subkey in subgroups.keys():
            df_group_two = dframe_grouped.loc[subgroups[subkey]].reset_index(drop=True)
            df_group_two["Label"] = 0
            var_al_wells = df_group_two["Metadata_well_position"].unique()
            df_sampled = pd.concat([df_group_one, df_group_two], ignore_index=True)

            if len(ref_al_wells) < 4:
                # ref_al_wells = np.random.choice(ref_al_wells, size=4)
                err_logger.write(f"{key}, {subkey} pair DOES NOT enough ref. alleles! Ref. allele wells in parquet: {ref_al_wells}\n")
                continue
            if len(var_al_wells) < 4:
                # var_al_wells = np.random.choice(var_al_wells, size=4)
                err_logger.write(f"{key}, {subkey} pair DOES NOT enough var. alleles! Var. allele wells in parquet: {var_al_wells}\n")
                continue
                
            well_pair_list = list(zip(ref_al_wells, var_al_wells))
            well_pair_nested_list = [[well_pair_list[i]] + well_pair_list[:i] + well_pair_list[i+1:] for i in range(len(well_pair_list))]
            ## try run classifier
            try:
                ## Bind df_sampled to the helper function
                classify_by_well_pair_bound = partial(classify_by_well_pair_exp_helper, df_sampled)
                result = thread_map(classify_by_well_pair_bound, well_pair_nested_list)
                
                pred_list = []
                for res in result:
                    if len(list(res.values())[0]) == 3:
                        feat_list.append(list(res.values())[0][0])
                        group_list.append(key)
                        pair_list.append(f"{key}_{subkey}")
                        info_list.append(list(res.values())[0][1])
                        pred_list.append(list(res.values())[0][2])
                    else:
                        print("res length not 3!")
                        feat_list.append([None] * len(feat_cols))
                        group_list.append(key)
                        pair_list.append(f"{key}_{subkey}")
                        info_list.append([None] * 10)

                cell_preds = pd.concat(pred_list, axis=0)
                cell_preds["Metadata_Protein"] = protein
                cell_preds["Metadata_Control"] = False
                table = pa.Table.from_pandas(cell_preds, preserve_index=False)
                pq_writer.write_table(table)
            except Exception as e:
                print(e)
                err_logger.write(f"{key}, {sub_key} error: {e}")

    ### Store feature importance
    df_feat_one = pd.DataFrame({"Group1": group_list, "Group2": pair_list})
    df_feat_two = pd.DataFrame(feat_list)
    df_feat = pd.concat([df_feat_one, df_feat_two], axis=1)
    df_feat["Metadata_Protein"] = protein
    df_feat["Metadata_Control"] = False

    # process classifier info
    df_result = pd.concat(info_list, ignore_index=True)
    df_result["Metadata_Control"] = False

    err_logger.write(f"Logging errors when running real experiments w/ protein {protein} finished.\n")
    err_logger.write(f"===========================================================================\n\n")
    return df_feat, df_result


def run_classify_workflow_rs(
    input_path: str,
    feat_output_path: str,
    info_output_path: str,
    preds_output_path: str,
    cc_threshold: int
    # use_gpu: Union[str, None] = "6,7",
):
    """
        Run modified workflow for single-cell classification
    """
    # if use_gpu is not None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu

    # Initialize parquet for cell-level predictions
    if os.path.exists(preds_output_path):
        os.remove(preds_output_path)
    schema = pa.schema([
        ("Classifier_ID", pa.string()),
        ("CellID", pa.string()),
        ("Label", pa.int64()),
        ("Prediction", pa.float32()),
        ("Metadata_Protein", pa.bool_()),
        ("Metadata_Control", pa.bool_()),
    ])
    writer = pq.ParquetWriter(preds_output_path, schema, compression="gzip")
    err_logger = io.StringIO()

    # Add CellID column
    dframe = (
        pl.scan_parquet(input_path)
        .with_columns(
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

    # Filter out wells with fewer than the cell count threhsold
    print("Filtering out ctrl wells if necessary:")
    df_control = drop_low_cc_wells(df_control, cc_threshold)
    print("Filtering out exp. wells (variant and reference alleles) if necessary:")
    df_exp = drop_low_cc_wells(df_exp, cc_threshold)

    err_logger = io.StringIO()
    # Protein feature analysis
    df_feat_pro_con, df_result_pro_con = control_group_runner_fewer_rep(
        df_control, pq_writer=writer, err_logger=err_logger, protein=True
    )
    df_feat_pro_exp, df_result_pro_exp = experimental_runner_plate_rep(
        df_exp, pq_writer=writer, err_logger=err_logger, protein=True
    )

    # Non-protein feature analysis
    df_feat_no_pro_con, df_result_no_pro_con = control_group_runner_fewer_rep(
        df_control, pq_writer=writer, err_logger=err_logger, protein=False
    )
    df_feat_no_pro_exp, df_result_no_pro_exp = experimental_runner_plate_rep(
        df_exp, pq_writer=writer, err_logger=err_logger, protein=False
    )
    writer.close()

    output_path = "/".join(feat_output_path.split("/")[:-1])
    # Save buffer contents to a file
    with open(f"{output_path}/logfile.txt", "w") as file:
        file.write(err_logger.getvalue())
    err_logger.close()

    # Concatenate results for both protein and non-protein
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
    df_result = df_result.drop_duplicates()

    # Write out feature importance and classifier info
    df_feat.to_csv(feat_output_path, index=False)
    df_result.to_csv(info_output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Taking in .")
    parser.add_argument('--input_path', type=str, required=True, help='Input parquet file')
    parser.add_argument('--output_path', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    ## INPUT ARGUMENTS FOR CLASSIFICATION EXAMPLES
    # input_path = "../../../6.downstream_analysis_snakemake/outputs/batch_profiles/2024_10_28_Batch_9_confocal/profiles_tcdropped_filtered_var_mad_outlier_featselect.parquet"
    # output_path = "../../output/classify_reimplement/2024_10_28_Batch_9_confocal/profiles_tcdropped_filtered_var_mad_outlier_featselect"
    if not os.path.exists(args.output_path):
        print(f"{args.output_path} doesn't exist. Creating it...")
        os.makedirs(args.output_path)
    
    preds_output_path = f"{args.output_path}/predictions.parquet"
    feat_output_path = f"{args.output_path}/feat_importance.csv"
    info_output_path = f"{args.output_path}/classifier_info.csv"
    
    run_classify_workflow_rs(args.input_path, 
                             feat_output_path=feat_output_path, 
                             info_output_path=info_output_path, 
                             preds_output_path=preds_output_path, 
                             cc_threshold=CC_THRESHOLD)


if __name__ == '__main__':
    main()