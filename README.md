# Overview

The [Mapping Africa](mappingafrica.io) project has developed models and data geared towards field boundary mapping in various countries in Africa. The largest of these efforts was through the Lacuna Fund-based project led by [Farmerline](https://farmerline.co/) and collaboration with [Spatial Collective](https://spatialcollective.com/) to develop [A Region-Wide, Multi-Year Set of Field Boundary Labels for Africa](https://github.com/agroimpacts/lacunalabels) (the Lacuna labels), which are now hosted on both the [Registry of Open Data on AWS](https://registry.opendata.aws/africa-field-boundary-labels/) and [Zenodo](https://zenodo.org/records/11060871).

In addition to that dataset, an additional set of \~5000 labels collected through various Mapping Africa project activities are to be made available. These data were combined with the Lacuna labels to train a modified U-Net model (described in [Khallaghi et al, 2025](https://www.mdpi.com/2072-4292/17/3/474)) that has been applied to map several countries, including Zambia, Tanzania, Angola, Ghana, and Nigeria.

The goal of this project is to integrate these datasets and models with the broader [Fields of the World](https://github.com/fieldsoftheworld) project. This integration will revolve around two primary efforts:

1.  Integrate the Lacuna+ labels with the existing FTW labels.
2.  Train and evaluate models using various combinations the integrated datasets. Models will include the existing Mapping Africa U-Net as well as FTW's U-Net variant, and potentially others.

## Set-up

We require `ftw-tools` to be installed (currently part of [ftw-baselines](https://github.com/fieldsoftheworld/ftw-baselines?tab=readme-ov-file#download-the-ftw-baseline-dataset)), which requires python 3.10-3.12. Using `pyenv` to manage and set up the environment:

``` bash
pyenv install -v 3.12.10
pyenv virtualenv 3.12.10 ftw-mapafrica
pyenv activate ftw-mapafrica
python -m pip install --upgrade pip
```

And then run `pip install -e .` to install the package in editable mode.

## Datasets

We retrieved the Mapping Africa/Lacuna+ labels from our own HPC storage, and the FTW dataset using the FTW cli.

``` bash
ftw download 
ftw data download -o ~/data/labels/cropland/
```

The Mapping Africa/Lacuna+ (hereafter MA) labels were resampled to 256x256. The image bands were re-ordered to RGB-NIR from their existing BGR-NIR order to be consistent with FTW. The 3-class label masks were reprocessed to have a 1-pixel wide field edge class, in keeping with the FTW labels.

A unified [CSV catalog](data/ftw-mappingafrica-combined-catalog.csv) was created that provides the following information:

| Variable    | Description                                                                                                                                                                                                                   |
|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| name        | FTW AOI ID and Lacuna+ grid identifier                                                                                                                                                                                        |
| dataset     | ftw or mappingafrica                                                                                                                                                                                                          |
| version     | 1.0 for FTW; 1.3.0 for labels collected under other Mapping Africa projects; 2.0.0 for labels from Lacuna Fund project                                                                                                        |
| country     | Full names for FTW, abbreviations for MA                                                                                                                                                                                      |
| x           | Longitude in decimal degrees                                                                                                                                                                                                  |
| y           | Latitude in decimal degrees                                                                                                                                                                                                   |
| fld_prop    | Proportion of image covered by field classes (interior + edge)                                                                                                                          |
| nonfld_prop | Proportion of image covered by non-field/background class                                                                                                                               |
| null_prop   | Proportion of image covered by unknown (3) class (FTW only)                                                                                                                             |
| window_a    | Partial path and name for image collected during the local dry season/end of season time period under the FTW scheme. This is the only time image time point available at present for MA.                                     |
| window_b    | Path and name of early growing season image for FTW, not available for MA.                                                                                                              |
| mask        | Path and name of 3-class mask. For FTW, the mask filename (as with image names) is formed from the AOI ID. For MA, the mask file name consists of `<name>*<assignment_id>*<year>-<month>.*` (year/month match window_a image). |
| split       | train, validate, or test                                                                                                                                                                                                      |

The image and mask names have partial paths to each that provide the respective sub-folder structures particular to each dataset, i.e.:

-   For FTW:

    -   images: `ftw/<country>/s2_images/<window_a|window_b>`

    -   masks: `ftw/<country>/label_masks/semantic_3class`

-   MA is simpler: `mappingafrica-256/<images|labels>`

So the two datasets should be downloaded into a single common folder to facilitate their combined use.

To access the MA dataset, download it using the AWS CLI:

``` bash
cd /path/to/your/common/label/directory
aws s3 sync s3://africa-field-boundary-labels/mappingafrica-256/ . --dryrun
aws s3 sync s3://africa-field-boundary-labels/mappingafrica-256/ .
```

If the –dryrun variant shows a successful download, run the final line to download the data into the same folder holding the FTW dataset.

## Working with the data

Data classes based on those in `ftw-baselines` and `torchgeo` are used here, based on those used in `ftw-baselines`, with modifications to provide additional augmentations and to read from the combined [catalog file](data/ftw-mappingafrica-combined-catalog.csv). See the [data-modules.ipynb](notebooks/data-modules.ipynb) for additional details. 

## Training and evaluation

From the CLI, the model can be trained as follows:

```bash
ftw_ma model fit -c configs/<config-file>.yaml
```

See the [example config](configs/example-config.yaml) for settings.

<config-file>.yaml should be named to be informative of the experiment, e.g. `fullcat-ftwbaseline-exp2.yaml` for the second experiment using the full combined catalog and FTW Baseline model.

To resume training from a specific checkpoint:

```bash
CKPT=/path/to/checkpoint/checkpoint.ckpt
ftw_ma model fit -c configs/<config-file>.yaml --ckpt_path $CKPT
```

To test the model:

```bash
CHKPT=/path/to/checkpoint/checkpoint.ckpt 
ftw_ma model test -c configs/config.yaml -m $CHKPT --gpu 0 -o metrics.json
```

Or run the `tester.sh` script:

```bash
# from project root
./scripts/tester.sh <model_dir_name> <version_number> <catalog>
```

<moidel_dir_name> is the name of the model directory under `~/working/models/`, e.g. `fullcat-ftwbaseline-exp2` where the model checkpoint is stored. <version_number> is the version number of the training run, specified explicitly as an integer, or if left empty the latest version is found and run. <catalog> is the path to the catalog CSV file, e.g. `data/ftw-mappingafrica-combined-catalog.csv`. 

This will produce an output metrics file in a specified directory, with a file name composed of the experiment name and catalog used in testing. The script is currently hard-coded for the validation split.

