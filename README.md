# MONAI GBM Segmentation

This repository contains the official implementation for the MIE 2025 publication:  
**"Automatic Segmentation of Histopathological Glioblastoma Whole-Slide Images utilizing MONAI"**

It also serves as a practical starting point for applying computer vision techniques to H&E-stained histopathology images using the [MONAI](https://monai.io/) framework. The code is **not GBM-specific** and can be reused for any binary segmentation task on tiled H&E images.

The project is configured for **binary segmentation**. To adapt it for multi-class tasks, update the `training.py` and `testing.py` modules as well as the loss function and metrics. To use the pipeline on non-H&E images, remove the H&E normalization.

---- 
## Usage Overview
The main entrypoint for running the code is the command-line script `monai_segmenter.py`.  
It was successfully tested on Windows, Linux, and macOS (limited test coverage – bugs may occur).

Use the `-h` flag to display available options for each mode:
```bash
python monai_segmenter.py train -h
```

---

## Setup
We highly recommend the creation of a new Python virtual environment with Python 3.10 (other versions were not tested and may require updates to the code/requirements).

The requirements can be installed via pip by running  `pip install -r requirements.txt`.

---

### Training Data Setup
The model is by default configured to run on binary segmentation on binary labels. Pixels with a value of 1 represent tumor tissue, while all other tissue is labeled with 0. All images should have an input size of 1024 x 1024 pixels and 3 color channels. It is highly recommended to work with `.png` files to allow for reproducible results.

The images should already be split into 3 folders `train`, `test`, and `validate` with the subfolders `img` and `lbl` in each of those. 

Besides the data set a source image for HE-stain normalization is required. This image is used to source the normalizer and adjust all other images to fit the provided image better. For this task a representative tile without artifacts and staining/morphological abnormalities. 

Summarized the model expects:
- **1024×1024** pixel RGB `.png` tiles
- **Binary labels** where tumor pixels = `1` and background = `0`
- Three main folders: `train`, `val`, and `test`, each with subfolders `img/` and `lbl/`
- One representative tile as reference for H&E normalization

This results in the following structure:
#### Expected Folder Structure
```
dataset/
├── train/
│   ├── img/
│   └── lbl/
├── val/
│   ├── img/
│   └── lbl/
├── test/
│   ├── img/
│   └── lbl/
normalizer_tile.png
```

### CLI Parameters

These are the most important CLI arguments (more via `-h`):

- `--train_image_path`, `--val_image_path`, `--test_image_path`: Folder with `img/` and `lbl/`
- `--normalizer_image_path`: Path to the reference tile for H&E normalization
- `--model_path`: Path to save or load a model checkpoint
- `--batch_size`: Training or testing batch size (default: 4)
- `--mode`: `"gpu"` or `"cpu"` (default: `"gpu"`)
- `--devices`: Optional GPU IDs (e.g. `"0"`, `"0,1"`)
- `--num_workers`: Number of parallel DataLoader workers (default: 4)

---

### CLI Commands

### Train a Model
To run the training of a new model with the provided script call it with the `train`-hub. A help page is provided for setup with the `-h`-flag `python monai_segmenter.py train -h`.

#### Example Call Train
```bash
python monai_segmenter.py train \
  --train_image_path ./data/train \
  --val_image_path ./data/val \
  --normalizer_image_path ./data/reference_tile.png \
  --model_path ./models/unet.ckpt \
  --batch_size 4 \
  --mode gpu
```

### Evaluate a Model
To run the evaluation of an already trained model with the provided script call it with the `test`-hub. A help page is provided for setup with the `-h`-flag `python monai_segmenter.py test -h`.


#### Example Call Test
```bash
python monai_segmenter.py test \
  --test_image_path ./data/test \
  --normalizer_image_path ./data/reference_tile.png \
  --model_path ./models/unet.ckpt
```

### Export a Model to ONNX
To export a trained model with the provided script call it with the `export`-hub. A help page is provided for setup with the `-h`-flag `python monai_segmenter.py export -h`.


#### Example Call Export to ONNX
```bash
python monai_segmenter.py export \
  --model_path ./models/unet.ckpt \
  --mode gpu
```

---

## Code quality
This project uses:
- [`Black`](https://black.readthedocs.io/) for automatic formatting
- [`Ruff`](https://docs.astral.sh/ruff/) for static code analysis

The configuration is located in [`pyproject.toml`](./pyproject.toml).

### Check lint
Using `ruff check .` in the `src` folder runs ruff over all Python files in the project. This does not change the code and returns if there are any linting errors regarding the ruff configuration.

### Correct lint automatically
To apply the linting errors found with `ruff check .` you can run it with the `--fix` flag:  `ruff check . --fix.
This tries to automatically fix all errors found during the check.

### Format code
Finally, `black .` should be used for a final cleanup, as ruff does only contain a partial ruleset.

----
## Cite
If you use the code provided in this repository please cite following publication: 
A citation entry will be added once the publication is available.
```
bibtex
@article{your2025paper,
  title = {...},
  author = {...},
  journal = {...},
  year = {2025}
}
```