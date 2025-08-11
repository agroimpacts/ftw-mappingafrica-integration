# Integrating Mapping Africa labels/models with Fields of the World

The [Mapping Africa](mappingafrica.io) project has developed models and data geared towards field boundary mapping in various countries in Africa. Among the largest of these efforts was through involvement with the Lacuna Fund-based project led by [Farmerline](https://farmerline.co/) and collaboration with [Spatial Collective](https://spatialcollective.com/) to develop [A Region-Wide, Multi-Year Set of Field Boundary Labels for Africa](https://github.com/agroimpacts/lacunalabels) (the Lacuna labels), which are now hosted on both the [Registry of Open Data on AWS](https://registry.opendata.aws/africa-field-boundary-labels/) and [Zenodo](https://zenodo.org/records/11060871).   

In addition to that dataset, an additional set of ~5000 labels collected through various Mapping Africa project activities are to be made available.  These data were combined with the Lacuna labels to train a modified U-Net model (described in [Khallaghi et al, 2025](https://www.mdpi.com/2072-4292/17/3/474)) that has been applied to map several countries, including Zambia, Tanzania, Angola, Ghana, and Nigeria. 

The goal of this project is to integrate these datasets and models with the broader [Fields of the World](https://github.com/fieldsoftheworld) project. This integration will revolve around two primary efforts:

1. Integrate the Lacuna+ labels with the existing FTW labels.
2. Train and evaluate models using various combinations the integrated datasets. Models will include the existing Mapping Africa U-Net as well as FTW's U-Net variant, and potentially others.

## Set-up

We require `ftw-tools` to be installed (currently part of [ftw-baselines](https://github.com/fieldsoftheworld/ftw-baselines?tab=readme-ov-file#download-the-ftw-baseline-dataset)), which requires python 3.10-3.12. Using `pyenv` to manage and set up the environment:

```bash
pyenv install -v 3.12.10
pyenv virtualenv 3.12.10 ftw-mapafrica
pyenv activate ftw-mapafrica
python -m pip install --upgrade pip
```

And then run `pip install -r requirements.txt` to install the package in editable mode.

### Datasets
We'll get the Lacuna+ labels from our own HPC storage, and the FTW dataset using the FTW cli. 

```bash
ftw download 
ftw data download -o ~/data/labels/cropland/
```





