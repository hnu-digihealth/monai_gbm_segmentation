"""
Testing module for MONAI-based GBM segmentation.

This script loads a trained UNet model and evaluates it on a test dataset
using MONAI metrics such as Dice score and Mean IoU.
"""

# Python Standard Library
import logging
from pathlib import Path

# Third Party Libraries
import numpy as np
import torch
import torchstain
from monai.data import DataLoader, Dataset
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, ToTensor
from pytorch_lightning import Trainer
from torchvision.io import read_image

# Local Libraries
from src.helper_functions.machine_learning import UNetLightning
from src.helper_functions.preprocessing import HENormalization

# Setup logger
logger = logging.getLogger("Testing")

def setup_test_dataloader(
    test_image_path: Path,
    normalizer_image_path: Path,
    batch_size: int,
    num_workers: int
) -> tuple[DataLoader, list]:
    """
    Prepares the DataLoader for testing.

    Args:
        test_image_path (Path): Directory with 'img' and 'lbl' subfolders.
        normalizer_image_path (Path): Reference image for normalization.
        batch_size (int): Batch size.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple[DataLoader, list]: Test DataLoader and list of test file dictionaries.
    """
    # Collect image and label file paths
    test_images_path = test_image_path / "img"
    test_labels_path = test_image_path / "lbl"

    test_images = sorted([x for x in test_images_path.iterdir() if x.suffix == ".png" and not x.name.startswith(".")])
    test_labels = sorted([x for x in test_labels_path.iterdir() if x.suffix == ".png" and not x.name.startswith(".")])
    test_files = [{"img": img, "seg": seg} for img, seg in zip(test_images, test_labels)]

    # Initialize and fit the Reinhard stain normalizer
    normalizer = torchstain.normalizers.ReinhardNormalizer(method="modified", backend="torch")
    normalizer.fit(read_image(normalizer_image_path))

    # Compose transformations including normalization
    test_transforms = Compose([
        LoadImaged(keys=["img", "seg"], dtype=np.float32, ensure_channel_first=True),
        HENormalization(keys=["img"], normalizer=normalizer, method="reinhard"),
        EnsureChannelFirstd(keys=["img"]),
        ToTensor(dtype=np.float32),
    ])

    # Create MONAI dataset and apply transforms
    test_ds = Dataset(data=test_files, transform=test_transforms)

    # Sanity check: ensure transformed image shapes are as expected
    try:
        assert test_ds[0]["img"].shape == torch.Size([3, 1024, 1024])
        assert test_ds[0]["seg"].shape == torch.Size([1, 1024, 1024])
    except AssertionError:
        logger.error("Image transformation check failed. Exiting.")
        exit(1)

    # Create DataLoader from dataset
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
    )

    return test_loader, test_files

def test_model(
    test_image_path: Path,
    normalizer_image_path: Path,
    batch_size: int,
    num_workers: int,
    model_path: Path,
) -> None:
    """
    Evaluate a trained model on test data using MONAI metrics.

    Args:
        test_image_path (Path): Path to test image and label directories.
        normalizer_image_path (Path): Path to the reference image for H&E normalization.
        batch_size (int): Batch size for evaluation.
        num_workers (int): Number of subprocesses for DataLoader.
        model_path (Path): Path to the trained model checkpoint.
    """
    logger.info("Starting test run")
    logger.info(f"Test image path: {test_image_path}")
    logger.info(f"Normalizer image path: {normalizer_image_path}")
    logger.info(f"Model checkpoint path: {model_path}")

    # Prepare data and transforms
    test_loader, test_files = setup_test_dataloader(
        test_image_path, normalizer_image_path, batch_size, num_workers
    )

    # Instantiate model with metrics
    logger.info("Loading trained model from checkpoint")
    pl_model = UNetLightning(
        test_loader,
        test_files,
        metric=DiceMetric(
            include_background=False,
            reduction="mean",
            num_classes=2,
            ignore_empty=False,
        ),
        metric_iou=MeanIoU(include_background=False, reduction="mean", ignore_empty=False),
    )

    # Initialize PyTorch Lightning Trainer
    logger.info("Starting model evaluation")
    trainer = Trainer(devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.test(pl_model, test_loader, ckpt_path=model_path)
    logger.info("Test run completed successfully")
