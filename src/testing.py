# Python Standard Library
from pathlib import Path

# Third Party Libraries
import numpy as np
import torch
from torchvision.io import read_image
import torchstain
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ToTensor
)
from monai.data import Dataset, DataLoader
from monai.metrics import DiceMetric, MeanIoU
from pytorch_lightning import Trainer

# Local Libraries
from src.helper_functions.preprocessing import HENormalization
from src.helper_functions.machine_learning import UNetLightning


def test_model(
    test_image_path: Path,
    normalizer_image_path: Path,
    batch_size: int,
    num_workers: int,
    model_path: Path
) -> None:
    test_images_path = test_image_path / 'img'
    test_labels_path = test_image_path / 'lbl'

    # setup img/label dicts
    test_images = sorted([x for x in test_images_path.iterdir() if x.suffix == '.png' and not x.name.startswith('.')])
    test_labels = sorted([x for x in test_labels_path.iterdir() if x.suffix == '.png' and not x.name.startswith('.')])

    test_files = [{'img': img, 'seg': seg} for img, seg in zip(test_images, test_labels)]

    # setup HE-staining normalizer
    normalizer = torchstain.normalizers.ReinhardNormalizer(method='modified', backend='torch')
    normalizer.fit(read_image(normalizer_image_path))

    # setup transformation composition and create dataset + data loader
    test_transforms = Compose([
        LoadImaged(keys=['img', 'seg'], dtype=np.float32, ensure_channel_first=True),
        HENormalization(keys=['img'], normalizer=normalizer, method='reinhard'),
        EnsureChannelFirstd(keys=['img']),
        ToTensor(dtype=np.float32),
    ])

    # assert the images are correctly transformed
    try:
        assert(test_ds[0]['img'].shape == torch.Size([3, 1024, 1024]))
        assert(test_ds[0]['seg'].shape == torch.Size([1, 1024, 1024]))
    except AssertionError:
        print("Transformation of Images failed, make sure only images are forwarded to the pipeline")
        exit(1)

    # setup data loaders
    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available()
    )

    # Load the trained model from the checkpoint

    pl_model = UNetLightning(
        test_loader, test_files,
        metric=DiceMetric(include_background=False, reduction="mean", num_classes=2, ignore_empty=False),
        metric_iou=MeanIoU(include_background=False, reduction="mean", ignore_empty=False),
    )

    # Initialize the trainer
    trainer = Trainer(devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu")

    # Evaluate the model on the test data
    trainer.test(pl_model, test_loader, ckpt_path=model_path)
