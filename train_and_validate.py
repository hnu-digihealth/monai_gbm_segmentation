# Python Standard Library
import os
from pathlib import Path
import sys
import logging

# Third-Party Libraries
import numpy as np
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ToTensor, 
    RandRotate90d, RandFlipd, RandAdjustContrastd, 
    RandGaussianNoised, AsDiscreted,
)
from monai.networks.nets import UNet
from monai.data import DataLoader, Dataset
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
import torch
import torchstain
import torchvision
import pytorch_lightning as pl

# Local Libraries
from helper_functions.preprocessing import HENormalization
from helper_functions.machine_learning import UNetLightning
from helper_functions.cmd_parser  import setup_argparser


# set GPU if required
# os.environ["CUDA_VISIBLE_DEVICES"]="2"


# set common seed for monai operations
monai.utils.misc.set_determinism(seed=421337133742)


def main(
        data_path, 
        src_image_path,
        batch_size,
        num_workers
    ):
    """
    data path should contain folder tran and validate with img and refined_labels folder each
    """
    # [ ] add comments
    # [ ] add arg parse
    #     [x] add required paths as args
    #     [ ] make all file paths variable
    # [x] remove unnecessary dependencies
    # [x] move code to utils
    # [ ] format code
    # [ ] add type hints

    torch.set_float32_matmul_precision('medium')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data paths
    train_images_path = Path(os.path.join(data_path,'minimal','img'))
    train_labels_path = Path(os.path.join(data_path,'minimal', 'inverted_lbls'))
    validate_images_path = Path(os.path.join(data_path,'minimal','img'))
    validate_labels_path = Path(os.path.join(data_path,'minimal', 'inverted_lbls'))
    # train_images_path = Path(os.path.join(data_path,'train','img'))
    # train_labels_path = Path(os.path.join(data_path,'train', 'refined_labels'))
    # validate_images_path = Path(os.path.join(data_path,'validate','img'))
    # validate_labels_path = Path(os.path.join(data_path,'validate', 'refined_labels'))

    # setup img/label dicts
    train_images = sorted([x for x in train_images_path.iterdir() if x.suffix == '.png' and not x.startswith('.')])
    train_labels = sorted([x for x in train_labels_path.iterdir() if x.suffix == '.png' and not x.startswith('.')])
    val_images = sorted([x for x in validate_images_path.iterdir() if x.suffix == '.png' and not x.startswith('.')])
    val_labels = sorted([x for x in validate_labels_path.iterdir() if x.suffix == '.png' and not x.startswith('.')])

    train_images = [str(x) for x in train_images]
    train_labels = [str(x) for x in train_labels]
    val_images = [str(x) for x in val_images]
    val_labels = [str(x) for x in val_labels]

    train_files = [{'img': img, 'seg': seg} for img, seg in zip(train_images, train_labels)]
    val_files = [{'img': img, 'seg': seg} for img, seg in zip(val_images, val_labels)]

    # setup HE-staining normalizer
    normalizer = torchstain.normalizers.ReinhardNormalizer(method='modified', backend='torch')
    src_image = torchvision.io.read_image(src_image_path)
    normalizer.fit(src_image)

    # setup transformation compositions and create datasets
    train_transforms = Compose([
        LoadImaged(keys=['img', 'seg'], dtype=np.int16, ensure_channel_first=True),
        #AsDiscreted(keys=['seg'], threshold=100),
        RandAdjustContrastd(keys=['img'], prob=0.5, gamma=(0.7, 1.3)),
        RandGaussianNoised(keys=['img'], prob=0.5, mean=0.0, std=0.01),
        HENormalization(keys=['img'], normalizer=normalizer, method='reinhard'),
        EnsureChannelFirstd(keys=['img']),
        RandFlipd(keys=['img', 'seg'], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=['img', 'seg'], prob=0.5),
        ToTensor(dtype=np.float32),
    ])

    val_transforms = Compose([
        LoadImaged(keys=['img', 'seg'], dtype=np.float32, ensure_channel_first=True),
        HENormalization(keys=['img'], normalizer=normalizer, method='reinhard'),
        #AsDiscreted(keys=['seg'], threshold=100, dtype=np.int32),
        EnsureChannelFirstd(keys=['img']),
        ToTensor(dtype=np.float32),
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    # assert the images are correctly transformed
    try:
        assert(train_ds[0]['img'].shape == torch.Size([3, 1024, 1024]))
        assert(train_ds[0]['seg'].shape == torch.Size([1, 1024, 1024]))
        assert(val_ds[0]['img'].shape == torch.Size([3, 1024, 1024]))
        assert(val_ds[0]['seg'].shape == torch.Size([1, 1024, 1024]))
    except AssertionError:
        logging.error("Transformation of Images failed, make sure only images are forwarded to the pipeline")
        sys.exit(1)

    # setup data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available()
    )

    # model initialization
    model = UNet(
        spatial_dims=2,
        in_channels=3,  # 3 for RGB images
        out_channels=1,  # 1 for binary segmentation
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,  # Number of residual units in each layer
        norm=Norm.BATCH,  # Batch normalization
        dropout=0.1
    )

    pl_model = UNetLightning(model, val_loader, val_files)

    trainer = pl.Trainer(
        max_epochs=500,
        devices=1, accelerator="gpu",# if torch.cuda.is_available() else 'cpu',
        precision=32,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val_dice', mode='max', save_top_k=2, verbose=True),
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=16, mode='min', verbose=True)
        ]
    )

    # train the model
    trainer.fit(pl_model, train_loader, val_loader)


if __name__ == "__main__":
    parser = setup_argparser()
    args = parser.parse_args()
    main(args.data_path, args.src_image_path, args.batch_size, args.num_workers)
