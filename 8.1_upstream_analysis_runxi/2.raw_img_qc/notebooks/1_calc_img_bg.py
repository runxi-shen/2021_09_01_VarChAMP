import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import os
    import re
    import sys
    import glob
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import seaborn as sns
    from skimage.io import imread
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    sys.path.append("../scripts")
    from img_utils import letter_dict, channel_dict
    letter_dict_rev = {v: k for k, v in letter_dict.items()}
    channel_dict_rev = {v: k for k, v in channel_dict.items()}

    # letter_dict_rev
    channel_list = list(channel_dict_rev.values())[:-3]
    # channel_list
    return


if __name__ == "__main__":
    app.run()
