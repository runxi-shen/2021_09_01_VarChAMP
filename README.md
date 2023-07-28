# 2021_09_01_VarCHAMP

## Reproducibility

### Python environment

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
