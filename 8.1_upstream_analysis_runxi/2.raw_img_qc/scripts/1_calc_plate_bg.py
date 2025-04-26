## This script calculates the background per each plate in the target batch
## Author: Runxi Shen
## Example usage: python 1_calc_plate_bg.py --batch_list 2025_01_27_Batch_13,2025_01_28_Batch_14

import argparse
import os
import re
import glob
import numpy as np
import polars as pl
from skimage.io import imread
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from img_utils import letter_dict, channel_dict

letter_dict_rev = {v: k for k, v in letter_dict.items()}
channel_dict_rev = {v: k for k, v in channel_dict.items()}
TIFF_IMG_DIR = "../inputs/images/"


def process_tiff_img(tiff_image_path, qs=np.array([50,60,70,75,80,90,95,99])):
    """
        Process the tiff img and output its summary metrics
    """
    img = imread(tiff_image_path)
    # median = float(np.median(img))
    percentiles = dict(zip([f"perc_{q}" for q in qs], np.percentile(img, q=qs)))

    tiff_img_name = tiff_image_path.split("/")[-1]
    site = re.search(r"(?<=f)(\d{2})(?=p)", tiff_img_name.split('-')[0])[0]
    channel = channel_dict_rev[re.search(r"(?<=ch)(\d+)(?=sk)", tiff_img_name.split('-')[1])[0]]
    well_letter = letter_dict_rev[re.search(r'(?<=r)(\d{2})(?=c)', tiff_img_name.split('-')[0])[0]]
    well_num = re.search(r'(?<=c)(\d{2})(?=f)', tiff_img_name.split('-')[0])[0]
    well = f"{well_letter}{well_num}"

    img_metrics_dict = {"plate": tiff_image_path.split('/')[-3].split("__")[0], 
                        "img_path": tiff_image_path, 
                        "site": site, "channel": channel, "well": well}
    img_metrics_dict.update(percentiles)

    return img_metrics_dict


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--batch_list", help="batch to process")
    p.add_argument("--workers", type=int, default=256, help="Number of parallel workers")
    p.add_argument("--output_dir", type=str, default="../outputs/1.plate_bg_summary")
    args = p.parse_args()

    batches = args.batch_list.split(",")
    for batch in batches:
        # find all TIFFs
        paths = glob.glob(f"{TIFF_IMG_DIR}/{batch}/images/*/Images/*.tiff", recursive=True)
        records = []

        # process in parallel with a progress bar
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            futures = {exe.submit(process_tiff_img, p): p for p in paths}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing TIFFs"):
                rec = fut.result()
                records.append(rec)
        
        if not os.path.exists(os.path.join(args.output_dir, batch)):
            os.makedirs(os.path.join(args.output_dir, batch))

        df = pl.DataFrame(records)
        df.write_parquet(os.path.join(args.output_dir, batch, "plate_bg.parquet"))


if __name__ == "__main__":
    main()