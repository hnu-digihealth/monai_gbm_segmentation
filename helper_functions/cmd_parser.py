# Python Standard Library
from argparse import ArgumentParser


def setup_argparser() -> ArgumentParser:
    parser = ArgumentParser(
        prog='GBM-MONAI',
        description='Train a U-Net model for segmentation of GBM tissue.'
    )

    parser.add_argument(
        '-d', '--data_path',
        type=str,
        required=True,
        help='Path to the data folder containing train and validate folders.',
    )
    parser.add_argument(
        '-s', '--src_image_path',
        type=str,
        required=True,
        help='Path to the source image for normalization.',
    )
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=16,
        help='Batch size for training.',
    )
    parser.add_argument(
        '-w', '--num_workers',
        type=int,
        default=8,
        help='Number of workers for the data loader.',
    )
    
    return parser
