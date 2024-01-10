# VarChAMP: Variant Characterization across the Mendelian Proteome

## Documents

**GDrive folder (internal)**: [link](https://drive.google.com/drive/folders/1HwRhkZFeNnrUFBL2zBwcojTiS5TSElFQ)


## What's in this repo?

This repo contains the analysis scripts and notebooks for the VarChAMP project.
The data is stored in a separate repo, [`2021_09_01_VarChAMP-data`](https://github.com/broadinstitute/2021_09_01_VarChAMP-data), which is added as a submodule to this repo.
Profiles from all the plates are in `2021_09_01_VarChAMP-data/profiles`.
All levels of profiles downstream of the aggregation step in the pycytominer workflow are in that folder.

## How to use this repo?

1. Fork the repo
2. Clone the repo

    ```bash
    git clone git@github.com:<YOUR USER NAME>/2021_09_01_VarChAMP.git
    ```

3. Download the contents of the submodule

    ```bash
    git submodule update --init --recursive
    cd 2021_09_01_VarChAMP-data
    dvc pull
    git lfs pull
    ```

4. Install the conda environment within each folder before running the notebooks.
   We use [mamba](https://mamba.readthedocs.io/en/latest/) to manage the computational environment.
   To install mamba see [instructions](https://mamba.readthedocs.io/en/latest/installation.html).
   After installing mamba, execute the following to install and navigate to the environment:

    ```bash
    # First, install the conda environment
    mamba env create --force --file environment.yml

    # If you had already installed this environment and now want to update it
    mamba env update --file environment.yml --prune

    # Then, activate the environment and you're all set!
    environment_name=$(grep "name:" environment.yml | awk '{print $2}')
    mamba activate $environment_name
    ```

5. Run the notebooks
