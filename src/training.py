"""
Training module for MONAI-based GBM segmentation.

This script sets up data pipelines, applies H&E normalization, defines
data augmentation, and trains a UNet model using PyTorch Lightning.
"""

# Python Standard Library
import logging
import time
from datetime import timedelta
from pathlib import Path
from sys import exit
from typing import Optional

# Third Party Libraries
import numpy as np
import torch
import torchstain
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    ToTensor,
)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchvision.io import read_image

# Local Libraries
from src.helper_functions.machine_learning import UNetLightning
from src.helper_functions.preprocessing import HENormalization

# Setup logger for training run
logger = logging.getLogger("Training")


def train_and_validate_model(
    train_image_path: Path,
    val_image_path: Path,
    normalizer_image_path: Path,
    batch_size: int,
    mode: str,
    devices: Optional[list[int]],
    model_path: Path,
    num_workers: int,
) -> None:
    """
    Train and validate the UNet segmentation model.

    Args:
        train_image_path (Path): Path to the training images and labels.
        val_image_path (Path): Path to the validation images and labels.
        normalizer_image_path (Path): Path to the reference image for stain normalization.
        batch_size (int): Batch size used during training.
        mode (str): Precision mode ('16-mixed' or '32').
        devices: Optional[list[int]] List of GPU device IDs or None to use CPU.
        model_path (Path): Directory to save model checkpoints.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        None
    """
    logger.info("Starting training run")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of workers: {num_workers}")
    logger.info(f"Precision: {mode}")
    logger.info("Loss function: DiceFocalLoss")
    logger.info("Optimizer: Adam (default Lightning)")
    logger.info(f"Train image path: {train_image_path}")
    logger.info(f"Validation image path: {val_image_path}")
    logger.info(f"Model checkpoint path: {model_path}")

    # Define paths to image and label subdirectories
    # TODO this can easily be moved to utils
    train_images_path = train_image_path / "img"
    train_labels_path = train_image_path / "lbl"
    validate_images_path = val_image_path / "img"
    validate_labels_path = val_image_path / "lbl"

    # Collect all .png images (ignoring hidden files)
    train_images = sorted([x for x in train_images_path.iterdir() if x.suffix == ".png" and not x.name.startswith(".")])
    train_labels = sorted([x for x in train_labels_path.iterdir() if x.suffix == ".png" and not x.name.startswith(".")])
    validate_images = sorted(
        [x for x in validate_images_path.iterdir() if x.suffix == ".png" and not x.name.startswith(".")]
    )
    validate_labels = sorted(
        [x for x in validate_labels_path.iterdir() if x.suffix == ".png" and not x.name.startswith(".")]
    )

    train_files = [{"img": img, "seg": seg} for img, seg in zip(train_images, train_labels)]
    val_files = [{"img": img, "seg": seg} for img, seg in zip(validate_images, validate_labels)]
    if not train_files:
        logger.warning("No training samples found - check your input folder.")
    if not val_files:
        logger.warning("No validation samples found - validation will be skipped or fail.")

    logger.info(f"Found {len(train_files)} training samples")
    logger.info(f"Found {len(val_files)} validation samples")
    logger.info("Initializing HE normalizer")

    # Initialize stain normalizer
    # TODO we use this multiple times -> move to function
    normalizer = torchstain.normalizers.ReinhardNormalizer(method="modified", backend="torch")
    normalizer.fit(read_image(normalizer_image_path))

    logger.info("Setting up training and validation transformations")

    # Define MONAI transforms for preprocessing and augmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"], dtype=np.int16, ensure_channel_first=True),
            RandAdjustContrastd(keys=["img"], prob=0.5, gamma=(0.7, 1.3)),
            RandGaussianNoised(keys=["img"], prob=0.5, mean=0.0, std=0.01),
            HENormalization(keys=["img"], normalizer=normalizer, method="reinhard"),
            EnsureChannelFirstd(keys=["img"]),
            RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["img", "seg"], prob=0.5),
            ToTensor(dtype=np.float32),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"], dtype=np.float32, ensure_channel_first=True),
            HENormalization(keys=["img"], normalizer=normalizer, method="reinhard"),
            EnsureChannelFirstd(keys=["img"]),
            ToTensor(dtype=np.float32),
        ]
    )

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    # assert the images are correctly transformed
    try:
        assert train_ds[0]["img"].shape == torch.Size([3, 1024, 1024])
        assert train_ds[0]["seg"].shape == torch.Size([1, 1024, 1024])
        assert val_ds[0]["img"].shape == torch.Size([3, 1024, 1024])
        assert val_ds[0]["seg"].shape == torch.Size([1, 1024, 1024])
    except AssertionError:
        print("Transformation of Images failed, make sure only images are forwarded to the pipeline")
        logger.error("Transformation of images failed. Check preprocessing pipeline.")
        exit(1)

    logger.info("Creating DataLoaders")

    # setup data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
    )

    # Define model, callbacks and trainer
    logger.info("Initializing UNetLightning model")
    pl_model = UNetLightning(val_loader, val_files)

    trainer = Trainer(
        #        max_epochs=500,
        max_epochs=5,
        devices=1,
        accelerator=mode,
        precision=32,
        callbacks=[
            ModelCheckpoint(monitor="val_dice", mode="max", save_top_k=2, verbose=True),
            EarlyStopping(monitor="val_loss", patience=16, mode="min", verbose=True),
        ],
        default_root_dir=model_path,
    )

    # train the model
    logger.info("Starting training")
    logger.info(f"Max epochs: {trainer.max_epochs}")
    start_time = time.time()
    trainer.fit(pl_model, train_loader, val_loader)
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = str(timedelta(seconds=int(duration)))
    logger.info(f"Training duration: {formatted_duration}")

    # Dynamically get best metric (if available)
    best_dice = trainer.callback_metrics.get("val_dice")
    if best_dice is not None:
        logger.info(f"Best validation Dice score: {best_dice:.4f}")
    else:
        logger.warning("No Dice score found in trainer callback metrics.")

    # Log model checkpoint path (based on ModelCheckpoint callback)
    checkpoints = trainer.checkpoint_callback.best_model_path
    if checkpoints:
        logger.info(f"Best model checkpoint saved to: {checkpoints}")
    else:
        logger.warning("No checkpoint was saved during training.")
    logger.info("Training completed")
