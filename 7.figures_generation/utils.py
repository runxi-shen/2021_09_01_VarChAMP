import warnings
from itertools import chain
import pandas as pd
import numpy as np 

warnings.filterwarnings("ignore")

def filter_common_genes(
    sc_profiles: pd.DataFrame,
    node_type_col: str = "Metadata_node_type",
    gene_col: str = "Metadata_symbol", 
    var_col: str = "Metadata_gene_allele"
):
    var_gene = list(sc_profiles[sc_profiles[node_type_col] == "allele"][gene_col].unique())
    var_num = sc_profiles[sc_profiles[node_type_col] == "allele"][var_col].unique().size
    
    ref_gene = list(sc_profiles[sc_profiles[node_type_col] == "disease_wt"][gene_col].unique())
    com_gene = list(set(var_gene)&set(ref_gene))
    print(f'This batch contains {var_num} variants from {len(var_gene)} genes, and {len(ref_gene)} references. The overlap between ref and var is {len(com_gene)}.')

    sc_profiles_filtered = sc_profiles[sc_profiles[gene_col].isin(com_gene)]
    var_num_filtered = sc_profiles_filtered[sc_profiles_filtered["Metadata_node_type"] == "allele"][var_col].unique().size
    print(f'After filtering, there are {var_num_filtered} variants left.')

    return sc_profiles_filtered, var_num_filtered

def split_result_dataframes(
        df_result: pd.DataFrame,
        score_col: str = 'PR_AUC',
):
    print(f'Dropped {df_result[df_result["PR_AUC"].isna()].shape[0]} NaN rows for {score_col} column.')
    df_result = df_result[~df_result[score_col].isna()]

    df_protein_ctrl = df_result[(df_result['Metadata_Protein']) & (df_result['Metadata_Control'])]
    df_protein_var = df_result[(df_result['Metadata_Protein']) & (~df_result['Metadata_Control'])]
    df_non_protein_ctrl = df_result[~(df_result['Metadata_Protein']) & (df_result['Metadata_Control'])]
    df_non_protein_var = df_result[~(df_result['Metadata_Protein']) & ~(df_result['Metadata_Control'])]

    return df_protein_ctrl, df_protein_var, df_non_protein_ctrl, df_non_protein_var

def take_median(df_protein_ctrl, df_protein_var, df_non_protein_ctrl, df_non_protein_var):
    df_protein_ctrl['Metadata_Platemap'] = df_protein_ctrl['Metadata_Plate'].apply(lambda x: x.split('T')[0])
    df_non_protein_ctrl['Metadata_Platemap'] = df_non_protein_ctrl['Metadata_Plate'].apply(lambda x: x.split('T')[0])
    
    df_protein_var = df_protein_var.groupby(['Group1', 'Group2']).agg({'PR_AUC':np.median}).reset_index()
    df_non_protein_var = df_non_protein_var.groupby(['Group1', 'Group2']).agg({'PR_AUC':np.median}).reset_index()
    df_protein_ctrl = df_protein_ctrl.groupby(['Group1', 'Metadata_Platemap', 'Group2']).agg({'PR_AUC':np.median}).reset_index()
    df_non_protein_ctrl = df_non_protein_ctrl.groupby(['Group1', 'Metadata_Platemap', 'Group2']).agg({'PR_AUC':np.median}).reset_index()

    df_protein_var['Metadata_Control'] = False
    df_non_protein_var['Metadata_Control'] = False
    df_protein_ctrl['Metadata_Control'] = True
    df_non_protein_ctrl['Metadata_Control'] = True
    
    df_protein_ctrl.drop('Metadata_Platemap', axis=1, inplace=True)
    df_non_protein_ctrl.drop('Metadata_Platemap', axis=1, inplace=True)

    return df_protein_ctrl, df_protein_var, df_non_protein_ctrl, df_non_protein_var

def upsample_for_plotting(df_var, df_control):
    upsample_fac = df_var.shape[0]//df_control.shape[0]

    plot_csv = pd.concat(
        [pd.concat([df_control]*upsample_fac), df_var]
    ).reset_index(drop=True)

    return plot_csv

def calculate_threshold(
        df_protein_var,
        df_protein_ctrl,
        df_non_protein_var,
        df_non_protein_ctrl,
        threshold: float = 95
):
    thresh = threshold
    print(f'Total number of ref-var pairs: {df_protein_var.shape[0]}')
    print(f'Total number of control well pairs: {df_protein_ctrl.shape[0]}')

    thresh_protein = np.percentile(np.array(df_protein_ctrl["PR_AUC"]), thresh)
    thresh_nonprotein = np.percentile(
        np.array(df_non_protein_ctrl["PR_AUC"]), thresh
    )

    print(f"{thresh} percentil of protein control: {thresh_protein}")
    print(f"{thresh} percentil of non-protein control: {thresh_nonprotein}")

    propass = np.where(df_protein_var["PR_AUC"] > thresh_protein)[0].size
    nonpropass = np.where(df_non_protein_var["PR_AUC"] > thresh_nonprotein)[
        0
    ].size

    print(f"Number of variants passed threshold (protein): {propass}")
    print(f"Number of variants passed threshold (non-protein): {nonpropass}")

    return thresh_protein, thresh_nonprotein

def get_threshold(
        df_thresh: pd.DataFrame,
        batch_id: str,
        protein: bool
):
    if protein:
        thresh = df_thresh[df_thresh['Batch_ID']==batch_id]['score_thresh_protein'].item()
    else:
        thresh = df_thresh[df_thresh['Batch_ID']==batch_id]['score_thresh_non_protein'].item()

    return thresh

def get_positive_alleles(
        df_protein: pd.DataFrame,
        df_non_protein: pd.DataFrame,
        df_thresh: pd.DataFrame,
        batch_id: str,
        score_col: str = 'PR_AUC'
):
    protein_thresh = get_threshold(df_thresh, batch_id, True)
    non_protein_thresh = get_threshold(df_thresh, batch_id, False)

    protein_passed = df_protein.iloc[np.where(df_protein[score_col] > protein_thresh)].reset_index(drop=True)
    non_protein_passed = df_non_protein.iloc[np.where(df_non_protein[score_col] > non_protein_thresh)].reset_index(drop=True)

    return protein_passed, non_protein_passed

def split_threshold_dataframe(
        df_thresh: pd.DataFrame,
        batch_id_1: str,
        batch_id_2: str,
):
    protein_thresh_1 = df_thresh[df_thresh['Batch_ID']==batch_id_1]['score_thresh_protein'].item()
    protein_thresh_2 = df_thresh[df_thresh['Batch_ID']==batch_id_2]['score_thresh_protein'].item()
    non_protein_thresh_1 = df_thresh[df_thresh['Batch_ID']==batch_id_1]['score_thresh_non_protein'].item()
    non_protein_thresh_2 = df_thresh[df_thresh['Batch_ID']==batch_id_2]['score_thresh_non_protein'].item()
    return protein_thresh_1, protein_thresh_2, non_protein_thresh_1, non_protein_thresh_2

def create_venn(list1, list2):
    com = len(list(set(list1) & set(list2)))
    list1_uniq = len(set(list1) - set(list2))
    list2_uniq = len(set(list2) - set(list1))
    return com, list1_uniq, list2_uniq

def get_protein_features(dframe: pd.DataFrame, protein_feat: bool):
    """Helper function to get dframe containing protein or non-protein features"""
    feat_col = find_feat_cols(dframe)

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
    return feat_col

def find_feat_cols(lframe):
    return [col for col in lframe.columns if not col.startswith('Metadata_')]
    
def remove_nan_infs_columns(dframe: pd.DataFrame):
    """Remove columns with NaN and INF"""
    feat_cols = find_feat_cols(dframe)
    withnan = dframe[feat_cols].isna().sum()[lambda x: x > 0]
    withinf = (dframe[feat_cols] == np.inf).sum()[lambda x: x > 0]
    withninf = (dframe[feat_cols] == -np.inf).sum()[lambda x: x > 0]
    redlist = set(chain(withinf.index, withnan.index, withninf.index))
    dframe_filtered = dframe[[c for c in dframe.columns if c not in redlist]]
    return dframe_filtered