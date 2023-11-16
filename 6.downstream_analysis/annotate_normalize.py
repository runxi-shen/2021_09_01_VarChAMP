import pathlib
import pandas as pd
from pycytominer.cyto_utils.cells import SingleCells
from pycytominer import annotate, normalize, feature_select
# ignore mix type warnings from pandas
import warnings
warnings.filterwarnings("ignore")
import os
from tqdm import tqdm

def annotate_with_platemap(profile_path, platemap_path, output_file_path):
    profile = pd.read_parquet(profile_path, engine="pyarrow")
    platemap = pd.read_csv(platemap_path).copy()
    # Append 'Metadata_' to platemap column names 
    platemap.columns = [
            f"Metadata_{x}" if not x.startswith("Metadata_") else x
            for x in platemap.columns
        ]
    aligned_df = platemap.merge(profile, on=["Metadata_Plate", "Metadata_Well"], how="right")
    aligned_df.to_parquet(path=output_file_path, compression="gzip")
    print(f'\n Annotated profile saved at {output_file_path}')
    
# Setting file paths
data_dir = pathlib.Path("/dgx1nas1/storage/data/sam/profiles").resolve(strict=True)
result_dir = pathlib.Path("/dgx1nas1/storage/data/sam/processed")
result_dir.mkdir(exist_ok=True)

# Metadata Path
platemap = '2023_05_30_B1A1R1.csv'

# annotating merged single-cell profile with metadata and normalize
for file in tqdm(os.listdir(data_dir)):
    if not file.endswith(".parquet"): continue
    orig_file = pathlib.Path(data_dir / file).resolve(strict=True)
    anot_file = pathlib.Path(result_dir / (str(file.split('.')[0])+'_annotated.parquet'))
    norm_file = pathlib.Path(result_dir / (str(file.split('.')[0])+'_annotated_normalized.parquet'))
    if os.path.isfile(norm_file): continue
    
    annotate_with_platemap(orig_file, platemap, anot_file)

    # Whole plate normalization
    normalize(
        profiles=anot_file,
        features="infer",
        image_features=False,
        meta_features="infer",
        samples="all",
        method='mad_robustize',
        mad_robustize_epsilon=0,
        output_file=norm_file,
        output_type="parquet",
        compression_options="gzip",
    )
    
    print(f"\n Normalized profile saved in: {norm_file}")