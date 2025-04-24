# Python Standard Library
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
from src.logging.setup_logger import setup_logger


# Setup logger for test run
logger = setup_logger("Testing")


def test_model(
    test_image_path: Path,
    normalizer_image_path: Path,
    batch_size: int,
    num_workers: int,
    model_path: Path,
) -> None:
    logger.info("Starting test run")
    logger.info(f"Test image path: {test_image_path}")
    logger.info(f"Normalizer image path: {normalizer_image_path}")
    logger.info(f"Model checkpoint path: {model_path}")
    test_images_path = test_image_path / "img"
    test_labels_path = test_image_path / "lbl"

    # setup img/label dicts
    test_images = sorted([x for x in test_images_path.iterdir() if x.suffix == ".png" and not x.name.startswith(".")])
    test_labels = sorted([x for x in test_labels_path.iterdir() if x.suffix == ".png" and not x.name.startswith(".")])
    logger.info(f"Found {len(test_images)} test images")

    test_files = [{"img": img, "seg": seg} for img, seg in zip(test_images, test_labels)]

    # setup HE-staining normalizer
    logger.info("Initializing stain normalizer (Reinhard)")
    normalizer = torchstain.normalizers.ReinhardNormalizer(method="modified", backend="torch")
    normalizer.fit(read_image(normalizer_image_path))

    # setup transformation composition and create dataset + data loader
    logger.info("Setting up test transformations")
    test_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"], dtype=np.float32, ensure_channel_first=True),
            HENormalization(keys=["img"], normalizer=normalizer, method="reinhard"),
            EnsureChannelFirstd(keys=["img"]),
            ToTensor(dtype=np.float32),
        ]
    )

    # setup data loaders
    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
    )
    logger.info(f"DataLoader created with batch size {batch_size}")

    # assert the images are correctly transformed
    try:
        assert test_ds[0]["img"].shape == torch.Size([3, 1024, 1024])
        assert test_ds[0]["seg"].shape == torch.Size([1, 1024, 1024])
    except AssertionError:
        print("Transformation of Images failed, make sure only images are forwarded to the pipeline")
        logger.error("Image transformation check failed. Exiting.")
        exit(1)

    # Load the trained model from the checkpoint
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

    # Initialize the trainer
    logger.info("Starting model evaluation")
    trainer = Trainer(devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu")

    # Evaluate the model on the test data
    trainer.test(pl_model, test_loader, ckpt_path=model_path)
    logger.info("Test run completed successfully")
