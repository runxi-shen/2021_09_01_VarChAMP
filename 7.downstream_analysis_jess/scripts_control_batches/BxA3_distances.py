# The B4A3 and B6A3 batches contain 8 different controls, each repeated 48X. 
# There is one WT-VAR pair (ALK) (positive protein loc control)
# The objective here is to assess the influence of well position effect within plates. 
# Previous attempts at doing this with classifiers failed because all classifiers were able to almost perfectly separate cells (saturated signal). 
# Here we explicitly compute distance.

import pandas as pd
import polars as pl
import pathlib
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")


def main():
    
    feature_type = "_annotated_normalized_featselected"
    batch_list = ["B4A3R1", "B6A3R2"]
    run_name = 'ALK_WT_VAR'
    
    batch = batch_list[0]
    
    # this will be a for loop after debugging
    data_dir = pathlib.Path(f"/dgx1nas1/storage/data/jess/varchamp/sc_data/processed_profiles/{batch}")
    result_dir = pathlib.Path(f'/dgx1nas1/storage/data/jess/varchamp/sc_data/classification_results/{batch}/{run_name}')
    result_dir.mkdir(exist_ok=True)
    
    sc_profile_path = f"{data_dir}/{batch}{feature_type}.parquet"
    
    # Read in data - only look at ALK wells
    sc_profiles = pl.scan_parquet(sc_profile_path).filter(pl.col("Metadata_SYMBOL") == "ALK").collect()
    
    # Get all metadata variable names  
    feat_col = [i for i in sc_profiles.columns if "Metadata_" not in i]
    
    # there are >1000 cells per well
    sc_profiles.group_by("Metadata_Well").agg(pl.count()).head
    
    # randomly sample 100 cells per well
    sampled_profs = sc_profiles.filter(pl.int_range(0, pl.count()).shuffle().over("Metadata_Well") < 250)
    sampled_profs.group_by("Metadata_Well").agg(pl.count()).head
    sampled_profs.shape # now there are < 10,000 cells - a more reasonable number for pairwise distances
    
    sampled_profs = sampled_profs.with_row_index()
    meta_wells = sampled_profs.select(["index", "Metadata_Well", "Metadata_allele"])
    
    
    # Compute pairwise euclidean distances between all cells
    euclid_dist = euclidean_distances(StandardScaler().fit_transform(sampled_profs.select(feat_col).to_numpy()))
    
    euclid_long = pd.DataFrame(euclid_dist).reset_index().melt('index').dropna()
    euclid_long = pl.from_pandas(euclid_long)
    euclid_long = euclid_long.with_columns(pl.col("index").cast(pl.UInt32))
    euclid_long = euclid_long.with_columns(pl.col("variable").cast(pl.UInt32))
    
    # add well annotations
    euclid_long = euclid_long.join(meta_wells, on="index")
    euclid_long = euclid_long.rename({"Metadata_Well": "Well_1", "Metadata_allele": "Allele_1", "index": "Cell_1", "variable": "index", "value": "euclid_dist"})
    euclid_long = euclid_long.join(meta_wells, on="index")
    euclid_long = euclid_long.rename({"Metadata_Well": "Well_2", "Metadata_allele": "Allele_2", "index": "Cell_2"})
    
    # compute mean distance by pairs of wells
    mean_euclid_dist = euclid_long.group_by(["Well_1", "Well_2", "Allele_1", "Allele_2"]).agg(pl.col("euclid_dist").mean().alias("euclid_dist_mean"),
                                                                      pl.col("euclid_dist").std().alias("euclid_dist_std"),
                                                                      pl.col("euclid_dist").median().alias("euclid_dist_median"))
    
    # Compute pairwise cosine distances between all cells
    cosine_dist = cosine_distances(StandardScaler().fit_transform(sampled_profs.select(feat_col).to_numpy()))
    
    cosine_long = pd.DataFrame(cosine_dist).reset_index().melt('index').dropna()
    cosine_long = pl.from_pandas(cosine_long)
    cosine_long = cosine_long.with_columns(pl.col("index").cast(pl.UInt32))
    cosine_long = cosine_long.with_columns(pl.col("variable").cast(pl.UInt32))
    
    # add well annotations
    cosine_long = cosine_long.join(meta_wells, on="index")
    cosine_long = cosine_long.rename({"Metadata_Well": "Well_1", "Metadata_allele": "Allele_1", "index": "Cell_1", "variable": "index", "value": "cosine_dist"})
    cosine_long = cosine_long.join(meta_wells, on="index")
    cosine_long = cosine_long.rename({"Metadata_Well": "Well_2", "Metadata_allele": "Allele_2", "index": "Cell_2"})
    
    # compute mean distance by pairs of wells
    mean_cosine_dist = cosine_long.group_by(["Well_1", "Well_2", "Allele_1", "Allele_2"]).agg(pl.col("cosine_dist").mean().alias("cosine_dist_mean"),
                                                                      pl.col("cosine_dist").std().alias("cosine_dist_std"),
                                                                      pl.col("cosine_dist").median().alias("cosine_dist_median"))
    
    # join both sets of distance metrics into single dataframe
    dist_res = mean_euclid_dist.join(mean_cosine_dist, on = ["Well_1", "Well_2", "Allele_1", "Allele_2"])
    
    # write out results
    dist_res.write_csv(f"{result_dir}/{batch}_well_distances.csv")
    
    
if __name__ == '__main__':
    main()