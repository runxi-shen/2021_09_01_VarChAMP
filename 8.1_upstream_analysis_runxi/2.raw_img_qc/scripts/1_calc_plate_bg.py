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


def summarize_tiff_img(tiff_image_path):
    img = imread(tiff_image_path)
    arr = img.ravel()
    # cast to float64 for safe sum
    s = arr.sum(dtype=np.float64)
    ss = (arr.astype(np.float64) ** 2).sum()
    n = arr.size
    # integer images → bincount histogram
    # minlength covers full dynamic range e.g. 0–65535 for uint16
    hist = np.bincount(arr, minlength=65536)
    return n, s, ss, hist


def summarize_img_group(tiff_imgs, output_dict, workers=2):
    """
    Summarize a list of TIFF images in parallel, returning list of (n, s, ss, hist) tuples.
    """
    if type(tiff_imgs)==str:
        tiff_imgs = glob.glob(tiff_imgs, recursive=True)

    if (tiff_imgs):
        results = []
        # process in parallel with a progress bar
        with ProcessPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(summarize_tiff_img, path): path for path in tiff_imgs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Summarizing TIFFs"):
                results.append(fut.result())
        # reduce
        total_n    = sum(r[0] for r in results)
        total_sum  = sum(r[1] for r in results)
        total_sumsq= sum(r[2] for r in results)
        # element-wise sum of histograms
        total_hist = sum(r[3] for r in results)
        # mean & std
        mean = total_sum / total_n
        std  = np.sqrt(total_sumsq/total_n - mean**2)
        # median: find intensity bin where cumsum ≥ N/2
        cum = np.cumsum(total_hist)
        median = np.searchsorted(cum, total_n//2)

        output_dict.update({"median": median, "mean": mean, "std": std})
        return output_dict
    else:
        output_dict.update({"median": np.nan, "mean": np.nan, "std": np.nan})
        return output_dict


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--batch_list", help="batch to process")
    p.add_argument("--workers", type=int, default=256, help="Number of parallel workers")
    p.add_argument("--output_dir", type=str, default="../outputs/1.plate_bg_summary")
    args = p.parse_args()

    batches = args.batch_list.split(",")
    # print("Summarize the per-channel site info by plate:")
    # for batch in batches:
    #     # find all TIFFs
    #     paths = glob.glob(f"{TIFF_IMG_DIR}/{batch}/images/*/Images/*.tiff", recursive=True)
    #     records = []

    #     # process in parallel with a progress bar
    #     with ProcessPoolExecutor(max_workers=args.workers) as exe:
    #         futures = {exe.submit(process_tiff_img, p): p for p in paths}
    #         for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing TIFFs"):
    #             rec = fut.result()
    #             records.append(rec)
        
    #     if not os.path.exists(os.path.join(args.output_dir, batch)):
    #         os.makedirs(os.path.join(args.output_dir, batch))

    #     df = pl.DataFrame(records)
    #     df.write_parquet(os.path.join(args.output_dir, batch, "plate_site_channel.parquet"))

    # print("Summarize the per-channel plate-level summary statistics:")
    # for batch in batches:
    #     tiff_img_dict_mapper = []
    #     plates = os.listdir(f"{TIFF_IMG_DIR}/{batch}/images")
    #     for plate in tqdm(plates):
    #         for channel in channel_dict_rev.keys():
    #             channel_tiffs = glob.glob(f"{TIFF_IMG_DIR}/{batch}/images/{plate}/Images/*ch{channel}sk*.tiff", recursive=True)[:100]
    #             if channel_tiffs:
    #                 result_dict = {"plate": plate.split("__")[0], 
    #                             "channel": channel_dict_rev[channel]}
    #                 result_dict["channel"] = channel_dict_rev[channel]
    #                 tiff_img_dict_mapper.append((channel_tiffs, result_dict))
                    
    #     results_per_plate = []
    #     with ProcessPoolExecutor(max_workers=384) as exe:
    #         # submit each (tiff_imgs, output_dict) pair as separate job
    #         futures = [
    #             exe.submit(summarize_img_group, tiff_imgs, output_dict)
    #             for tiff_imgs, output_dict in tiff_img_dict_mapper
    #         ]
    #         for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing tiffs per plate"):
    #             result = fut.result()
    #             results_per_plate.append(result)

    #     df_plate = pl.DataFrame(results_per_plate)
    #     df_plate.write_parquet(os.path.join(args.output_dir, batch, "plate_sum_stats.parquet"))

    # print("Summarize the per-channel well-level summary statistics:")
    # for batch in batches:
    #     tiff_img_dict_mapper = []
    #     plates = os.listdir(f"{TIFF_IMG_DIR}/{batch}/images")
    #     for plate in plates:
    #         all_tiffs = glob.glob(f"{TIFF_IMG_DIR}/{batch}/images/{plate}/Images/*.tiff")
    #         unique_wells = sorted(set([tiff.split('/')[-1][:6] for tiff in all_tiffs]))
    #         print(len(unique_wells)) ## 384
    #         for well in tqdm(unique_wells):
    #             well_letter = letter_dict_rev[re.search(r'(?<=r)(\d{2})(?=c)', well)[0]]
    #             well_num = re.search(r'(?<=c)(\d{2})', well)[0]
    #             # print(well, result_dict["well"])
    #             for channel in channel_dict_rev.keys():
    #                 result_dict = {"plate": plate.split("__")[0], 
    #                             "well": f"{well_letter}{well_num}",
    #                             "channel": channel_dict_rev[channel]}
    #                 channel_tiffs = f"{TIFF_IMG_DIR}/{batch}/images/{plate}/Images/{well}*-ch{channel}sk*.tiff" # glob.glob(f"{TIFF_IMG_DIR}/{batch}/images/{plate}/Images/{well}*_ch{channel}sk*.tiff", recursive=True)[:100]
    #                 # print(channel_tiffs)
    #                 tiff_img_dict_mapper.append((channel_tiffs, result_dict))

    #     results_per_well = []
    #     with ProcessPoolExecutor(max_workers=384) as exe:
    #         # submit each (tiff_imgs, output_dict) pair as separate job
    #         futures = [
    #             exe.submit(summarize_img_group, tiff_imgs, output_dict)
    #             for tiff_imgs, output_dict in tiff_img_dict_mapper
    #         ]
    #         for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing tiffs per well"):
    #             result = fut.result()
    #             results_per_well.append(result)

    #     df = pl.DataFrame(results_per_well)
    #     df.write_parquet(os.path.join(args.output_dir, batch, "plate_well_sum_stats.parquet"))


if __name__ == "__main__":
    main()