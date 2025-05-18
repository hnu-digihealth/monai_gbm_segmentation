"""
Main entrypoint for the MONAI GBM segmentation CLI.

Parses command-line arguments and dispatches to training, testing, or ONNX export.
"""

# Python Standard Library
import logging
from argparse import Namespace

# Third Party Libraries
from monai.utils.misc import set_determinism
from torch import set_float32_matmul_precision

# Local Libraries
from src.argparser.argparser_setup import cli_core
from src.logging.setup_logger import setup_logger
from src.onnx_export import export_model
from src.testing import test_model
from src.training import train_and_validate_model


def main(args: Namespace, logger: logging.Logger) -> None:
    """
    Main pipeline dispatcher.

    Args:
        args (Namespace): Parsed CLI arguments.
        logger (Logger): Configured logger instance.

    Routes execution to the appropriate pipeline step based on `args.hub`.
    """
    logger.info("Starting MONAI Segmenter")
    logger.info(f"Selected mode: {args.hub}")
    logger.info(f"Seed: {args.seed}, Precision mode: medium")

    # set MONAI seed for all following random operations
    set_determinism(seed=args.seed)
    # set pytorch float32 matmul precision
    set_float32_matmul_precision("medium")

    match args.hub:
        case "train":
            logger.info("Launching training pipeline")
            train_and_validate_model(
                args.train_path,
                args.val_path,
                args.normalizer_image_path,
                args.batch_size,
                args.mode,
                args.devices,
                args.model_path,
                args.num_workers,
            )
        case "test":
            logger.info("Launching testing pipeline")
            test_model(
                args.test_path,
                args.normalizer_image_path,
                args.batch_size,
                args.num_workers,
                args.model_path,
                args.mode,
                args.devices,
            )
        case "export":
            logger.info("Launching ONNX export")
            export_model(
                args.model_path,
                args.test_path,
                args.normalizer_image_path,
            )


if __name__ == "__main__":
    parser = cli_core()
    args = parser.parse_args()
    logger = setup_logger("Segmenter", level=args.log_level)
    logger.info(f"Parsed CLI arguments: {args}")
    main(args, logger)
