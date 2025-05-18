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


def setup_dataloaders(
    train_image_path: Path, val_image_path: Path, normalizer_image_path: Path, batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader]:
    """
    Prepares DataLoaders for training and validation.

    Args:
        train_image_path (Path): Path to training image and label folders.
        val_image_path (Path): Path to validation image and label folders.
        normalizer_image_path (Path): Path to reference image used for stain normalization.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    # Create and fit the raw stain normalizer instance
    raw_normalizer = torchstain.normalizers.ReinhardNormalizer(method="modified", backend="torch")
    raw_normalizer.fit(read_image(normalizer_image_path))

    def prepare_dataset(img_dir: Path, lbl_dir: Path, transform_type: str) -> Dataset:
        """
        Prepares a MONAI Dataset for training or validation.

        Args:
            img_dir (Path): Directory containing images.
            lbl_dir (Path): Directory containing segmentation masks.
            transform_type (str): "train" or "val", to determine applied transforms.

        Returns:
            Dataset: A MONAI Dataset with the appropriate transforms.
        """
        # Collect images and segmentation masks
        images = sorted([x for x in img_dir.iterdir() if x.suffix == ".png" and not x.name.startswith(".")])
        labels = sorted([x for x in lbl_dir.iterdir() if x.suffix == ".png" and not x.name.startswith(".")])
        data_dicts = [{"img": img, "seg": seg} for img, seg in zip(images, labels)]

        # Define transformation pipeline depending on mode
        if transform_type == "train":
            transforms = Compose(
                [
                    LoadImaged(keys=["img", "seg"], dtype=np.int16, ensure_channel_first=True),
                    RandAdjustContrastd(keys=["img"], prob=0.5, gamma=(0.7, 1.3)),
                    RandGaussianNoised(keys=["img"], prob=0.5, mean=0.0, std=0.01),
                    HENormalization(keys=["img"], normalizer=raw_normalizer, method="reinhard"),
                    EnsureChannelFirstd(keys=["img"]),
                    RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=0),
                    RandRotate90d(keys=["img", "seg"], prob=0.5),
                    ToTensor(dtype=np.float32),
                ]
            )
        else:
            transforms = Compose(
                [
                    LoadImaged(keys=["img", "seg"], dtype=np.float32, ensure_channel_first=True),
                    HENormalization(keys=["img"], normalizer=raw_normalizer, method="reinhard"),
                    EnsureChannelFirstd(keys=["img"]),
                    ToTensor(dtype=np.float32),
                ]
            )

        return Dataset(data=data_dicts, transform=transforms)

    # Prepare datasets for training and validation
    train_ds = prepare_dataset(train_image_path / "img", train_image_path / "lbl", "train")
    val_ds = prepare_dataset(val_image_path / "img", val_image_path / "lbl", "val")

    # Check expected image and label shapes
    try:
        assert train_ds[0]["img"].shape == torch.Size([3, 1024, 1024])
        assert train_ds[0]["seg"].shape == torch.Size([1, 1024, 1024])
        assert val_ds[0]["img"].shape == torch.Size([3, 1024, 1024])
        assert val_ds[0]["seg"].shape == torch.Size([1, 1024, 1024])
    except AssertionError:
        logger.error("Transformation of images failed. Check preprocessing pipeline.")
        exit(1)

    # Create DataLoaders
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

    return train_loader, val_loader


def configure_trainer(model_path: Path, devices: Optional[list], mode: str) -> Trainer:
    """
    Configures the PyTorch Lightning Trainer.

    Args:
        model_path (Path): Directory to save checkpoints.
        devices (Optional[list[int]]): GPU device IDs or None to use CPU.
        mode (str): Accelerator type (e.g. "cpu", "gpu").

    Returns:
        Trainer: Configured PyTorch Lightning Trainer.
    """
    # Ensure devices is int or list[int], not str
    if isinstance(devices, str):
        try:
            devices = [int(devices)]
        except ValueError as err:
            raise ValueError(f"Invalid device string: {devices}") from err

    if mode == "cpu":
        devices = None

    return Trainer(
        max_epochs=150,
        devices=devices or -1,
        accelerator=mode,
        precision=32,
        callbacks=[
            ModelCheckpoint(monitor="val_dice", mode="max", save_top_k=2, verbose=True),
            EarlyStopping(monitor="val_loss", patience=16, mode="min", verbose=True),
        ],
        default_root_dir=model_path,
    )


def run_training(model: UNetLightning, trainer: Trainer, train_loader: DataLoader, val_loader: DataLoader) -> None:
    """
    Runs the training loop.

    Args:
        model (UNetLightning): Lightning module for UNet.
        trainer (Trainer): Configured Lightning Trainer.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
    """
    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    elapsed = timedelta(seconds=round(time.time() - start_time))
    logger.info(f"Training finished. Total time: {elapsed}")

    # Log best validation Dice score
    best_dice = trainer.callback_metrics.get("val_dice")
    if best_dice is not None:
        logger.info(f"Best validation Dice score: {best_dice:.4f}")
    else:
        logger.warning("No Dice score found in trainer callback metrics.")

    # Log path to best checkpoint
    checkpoints = trainer.checkpoint_callback.best_model_path
    if checkpoints:
        logger.info(f"Best model checkpoint saved to: {checkpoints}")
    else:
        logger.warning("No checkpoint was saved during training.")


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
    High-level pipeline for training and validating the segmentation model.

    Args:
        train_image_path (Path): Path to training data folder.
        val_image_path (Path): Path to validation data folder.
        normalizer_image_path (Path): Reference image path for stain normalization.
        batch_size (int): Batch size.
        mode (str): Accelerator mode (e.g., "gpu").
        devices (Optional[list[int]]): List of GPU device IDs or None.
        model_path (Path): Directory to save model checkpoints.
        num_workers (int): Number of DataLoader workers.
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

    # Prepare data
    logger.info("Setting up dataloaders...")
    train_loader, val_loader = setup_dataloaders(
        train_image_path, val_image_path, normalizer_image_path, batch_size, num_workers
    )

    # Initialize model and training configuration
    logger.info("Initializing model and trainer...")
    model = UNetLightning(val_loader, val_loader.dataset.data)
    trainer = configure_trainer(model_path, devices, mode)

    # Start training
    logger.info("Starting training...")
    run_training(model, trainer, train_loader, val_loader)
    logger.info("Training completed")
