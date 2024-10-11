# Python Standard Library
import os
from pathlib import Path
import sys
import logging

# Third-Party Libraries
import numpy as np
import monai
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ToTensor, AsDiscreted
from monai.networks.nets import UNet
from monai.data import DataLoader, Dataset
from monai.networks.layers import Norm
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
        num_workers,
        model_path
    ):  
    # setup data paths
    test_images_path = Path(os.path.join(data_path,'test','img'))
    test_labels_path = Path(os.path.join(data_path,'test', 'inverted_lbls'))

    # setup img/label dicts
    test_images = sorted([x for x in test_images_path.iterdir() if x.suffix == '.png'])
    test_labels = sorted([x for x in test_labels_path.iterdir() if x.suffix == '.png'])

    test_files = [{'img': img, 'seg': seg} for img, seg in zip(test_images, test_labels)]

    # setup HE-staining normalizer
    normalizer = torchstain.normalizers.ReinhardNormalizer(method='modified', backend='torch')
    src_image = torchvision.io.read_image(src_image_path)
    normalizer.fit(src_image)

    # setup transformation composition and create dataset + data loader
    test_transforms = Compose([
        LoadImaged(keys=['img', 'seg'], dtype=np.float32, ensure_channel_first=True),
        HENormalization(keys=['img'], normalizer=normalizer, method='reinhard'),
        #AsDiscreted(keys=['seg'], threshold=100, dtype=np.int32),
        EnsureChannelFirstd(keys=['img']),
        ToTensor(dtype=np.float32),
    ])

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available()
    )

    # assert the transformer is working correctly
    try:
        assert(test_ds[0]['img'].shape == torch.Size([3, 1024, 1024]))
        assert(test_ds[0]['seg'].shape == torch.Size([1, 1024, 1024]))
    except AssertionError:
        logging.error("Transformation of Images failed, make sure only images are forwarded to the pipeline")
        sys.exit(1)

    # Load the trained model from the checkpoint
    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=0.1
    )


    pl_model = UNetLightning(model, test_loader, test_files)

    # Initialize the trainer
    trainer = pl.Trainer(devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu")

    # Evaluate the model on the test data
    trainer.test(pl_model, test_loader, ckpt_path=model_path)


if __name__ == "__main__":
    parser = setup_argparser()
    parser.add_argument(
        '-m', '--model_path',
        type=str,
        required=True,
        help='Path to the trained model'
    )

    args = parser.parse_args()
    main(args.data_path, args.src_image_path, args.batch_size, args.num_workers, args.model_path)
