"""
CLI setup module for the MONAI GBM segmentation pipeline.

Defines subcommands for training, testing, and ONNX export,
each with their own argument parser setup.
"""

# Python Standard Library
from argparse import ArgumentParser, _SubParsersAction

# Local Libraries
from src.argparser.setup_monai_args import setup_monai_args
from src.argparser.setup_path_args import setup_path_args

# Global Variables
VERSION = "0.1.0"


def cli_core() -> ArgumentParser:
    """
    Create and return the main ArgumentParser for the CLI.

    Returns:
        parser (ArgumentParser): The root argument parser.
    """
    # Set description for cli core
    desc = """Command-line interface for MONAI Histo Segmenter: a simple showcase for histopathological image \
    segmentation with MONAI"""

    # Setup ArgumentParser interface
    parser = ArgumentParser(prog="MONAI Histo Segmenter", description=desc)

    # Add optional core arguments
    parser.add_argument("-v", "--version", action="version", version="%(prog)s_v" + VERSION)

    # Add subparser interface
    subparsers = parser.add_subparsers(
        title="Application Modes",
        description="Choose the mode to run the application in.",
        required=True,
        dest="hub",
    )

    __cli_train(subparsers)
    __cli_test(subparsers)
    __cli_onnx_export(subparsers)

    # Return parsers
    return parser


def __cli_train(subparsers: _SubParsersAction) -> None:
    """
    Add 'train' subcommand with path and MONAI-specific arguments.
    """
    desc = """ Pipeline hub for Training the MONAI model on a tiled dataset """

    parser_train = subparsers.add_parser("train", help=desc, add_help=False)

    setup_path_args(parser_train, "train")
    setup_monai_args(parser_train, "train")
    __setup__help_args(parser_train)


def __cli_test(subparsers: _SubParsersAction) -> None:
    """
    Add 'test' subcommand for evaluating a trained model.
    """
    desc = """ Pipeline hub for Testing a model trained with the training hub """

    parser_train = subparsers.add_parser("test", help=desc, add_help=False)

    setup_path_args(parser_train, "test")
    setup_monai_args(parser_train, "test")
    __setup__help_args(parser_train)


def __cli_onnx_export(subparsers: _SubParsersAction) -> None:
    """
    Add onnx 'export' subcommand for exporting a model to ONNX format.
    """
    desc = """ Pipeline hub for Exporting a model trained with the training hub """

    parser_train = subparsers.add_parser("export", help=desc, add_help=False)

    setup_path_args(parser_train, "export")
    __setup__help_args(parser_train)


def __setup__help_args(parser: ArgumentParser) -> None:
    """
    Add default help flag to each subcommand manually.
    """
    arg_group = parser.add_argument_group("Arguments - Other")
    arg_group.add_argument("-h", "--help", action="help", help="show this help message and exit")

    arg_group.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
