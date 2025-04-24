# MONAI GBM Segmentation
This repository provides the code for the MIE 2025 publication: "Automatic Segmentation of Histopathological Glioblastoma Whole-Slide Images utilizing MONAI".<br>
It can further be used as a getting started guide for computer vision on histopathological data with the MONAI framework. The Code is not Glioblastoma (GBM) specific an can be applied to all hematoxylin and eosin (HE)-stained histopathological segmentation tasks on tiled datasets.<br>
The project is set up for binary segmentation of HE tiles. For multi-class segmentation, changes have to be made in the `training.py` and `testing.py` and loss functions/metrics have to be updated. For non HE segmentation the HE normalizer has to be removed from the code.

---- 
## Usage
The provided code is mend to be run via the command line script `monai_segmenter.py`. The script was successfully executed on windows, linux, and mac. However, no extensive testing was conducted. Bugs may occur.

When starting the script with `-h` a help page is provided. 

### Setup
We highly recommend the creation of a new Python virtual environment with Python 3.10 (other versions were not tested and may require updates to the code/requirements).

The requirements can be installed via pip by running  `pip install -r requirements.txt`.

### Training Data Setup
The model is by default configured to run on binary segmentation on binary labels. Pixels with a value of 1 represent tumor tissue, while all other tissue is labeled with 0. All images should have an input size of 1024 x 1024 pixels and 3 color channels. It is highly recommended to work with `.png` files to allow for reproducible results. <br>
The images should already be split into 3 folders `train`, `test`, and `validate` with the subfolders `img` and `lbl` in each of those. 

Besides the data set a source image for HE-stain normalization is required. This image is used to source the normalizer and adjust all other images to fit the provided image better. For this task a representative tile without artifacts and staining/morphological abnormalities. 

### Run Training
To run the training of a new model with the provided script call it with the `train`-hub. A help page is provided for setup with the `-h`-flag `python monai_segmenter.py train -h`.

<!-- add something about the parameters and such-->

### Run Testing
To run the evaluation of an already trained model with the provided script call it with the `test`-hub. A help page is provided for setup with the `-h`-flag `python monai_segmenter.py test -h`.

<!-- add something about the parameters and such-->

### Export to ONNX
To export a trained model with the provided script call it with the `export`-hub. A help page is provided for setup with the `-h`-flag `python monai_segmenter.py export -h`.

<!-- add something about the parameters and such-->

## Code quality
This project uses [Black](https://black.readthedocs.io/) and [Ruff](https://docs.astral.sh/ruff/) for automatic formatting and static code analysis.  
The configuration is located in [`pyproject.toml`](./pyproject.toml).

### Check lint
Using `ruff check .` in the `src` folder runs ruff over all Python files in the project. This does not change the code and returns if there are any linting errors regarding the ruff configuration.

### Correct lint automatically
To apply the linting erros found with `ruff check .` you can run it with the `--fix` flag:  `ruff check . --fix.
This tries to automatically fix all errors found during the checkout.

### Format code
Finally, `black .` should be used for a final cleanup, as ruff does only contain a partial ruleset.

----
## Cite
If you use the code provided in this repository please cite following publication: 
```
bibtex
```