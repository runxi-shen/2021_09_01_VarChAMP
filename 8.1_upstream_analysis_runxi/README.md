# VarChAMP Pipeline Upstream Analyses for Data QC

__Auther: Runxi Shen__

In this subfolder, we examine the metadata for each genetic variant and the layouts of 384-well platemaps for screening them through Variant Painting imaging. 

## 1. metadata_qc

Download the platemaps information from cell-painting gallery from aws and the corresponding metadata of genetic variants provided by Taipale Lab at UofT, and cross-check them for data consistency and data formats.

If the original platemap information downloaded from aws needs further correction due to changes in experimental set-ups and inconsistent file formatting, etc., we need to correct for the platemap information in this directory and re-uploaded the files to the aws cell-painting gallery.

## 2. raw_img_qc

Check the imaging wells with extremely low GFP expressions, which are on the same level as the background and considered as technical noises, and flag them in the downstream analyses.

The procedure for flag the imaging wells:

1. For each imaging plate, we calculate the summary statistics of each imaging site, well, and the whole plate, such as their [25, 50, 75, 80, 85, 90, 95, 99]th percentiles, mean, and standard deviations;

2. For each imaging well, we use the log ratio of log10(99th perc / 25th perc) as its signal-to-noise ratio, and plot the distribution of the log-ratios for each plate. The first bin out of the 50 bins across the whole imaging plate are identified as "outlier" log-ratios. All the imaging wells from this first bin are considered to capture nothing but techinical noises, and no actual cell is identified.