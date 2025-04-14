# Python Standard Library
from pathlib import Path
from sys import exit

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

from src.helper_functions.machine_learning import UNetLightning

# Local Libraries
from src.helper_functions.preprocessing import HENormalization


def train_and_validate_model(
    train_image_path: Path,
    val_image_path: Path,
    normalizer_image_path: Path,
    batch_size: int,
    mode: str,
    devices: list | None,
    model_path: Path,
    num_workers: int,
):
    # setup data paths
    # TODO this can easily be moved to utils
    train_images_path = train_image_path / "img"
    train_labels_path = train_image_path / "lbl"
    validate_images_path = val_image_path / "img"
    validate_labels_path = val_image_path / "lbl"

    # setup img/label dicts
    train_images = sorted(
        [
            x
            for x in train_images_path.iterdir()
            if x.suffix == ".png" and not x.name.startswith(".")
        ]
    )
    train_labels = sorted(
        [
            x
            for x in train_labels_path.iterdir()
            if x.suffix == ".png" and not x.name.startswith(".")
        ]
    )
    validate_images = sorted(
        [
            x
            for x in validate_images_path.iterdir()
            if x.suffix == ".png" and not x.name.startswith(".")
        ]
    )
    validate_labels = sorted(
        [
            x
            for x in validate_labels_path.iterdir()
            if x.suffix == ".png" and not x.name.startswith(".")
        ]
    )

    train_files = [
        {"img": img, "seg": seg} for img, seg in zip(train_images, train_labels)
    ]
    val_files = [
        {"img": img, "seg": seg} for img, seg in zip(validate_images, validate_labels)
    ]

    # setup HE-staining normalizer
    # TODO we use this multiple times -> move to function
    normalizer = torchstain.normalizers.ReinhardNormalizer(
        method="modified", backend="torch"
    )
    normalizer.fit(read_image(normalizer_image_path))

    # setup transformations
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
            LoadImaged(
                keys=["img", "seg"], dtype=np.float32, ensure_channel_first=True
            ),
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
        print(
            "Transformation of Images failed, make sure only images are forwarded to the pipeline"
        )
        exit(1)

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

    # model initialization
    pl_model = UNetLightning(val_loader, val_files)

    trainer = Trainer(
        max_epochs=500,
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
    trainer.fit(pl_model, train_loader, val_loader)
