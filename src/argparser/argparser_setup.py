# Python Standard Library
from argparse import ArgumentParser

from src.argparser.setup_monai_args import setup_monai_args

# Local Libraries
from src.argparser.setup_path_args import setup_path_args

# Global Variables
VERSION = "0.1.0"


def cli_core():
    """
    Internal function for Command-Line-Interface (CLI) setup.

    Parameters
    ----------

    Returns
    -------
    - `parser: ArgumentParser` - The main argument parser.
    - `subparsers` - The subparser endpoint to attach new subparsers to.

    """
    # Set description for cli core
    desc = """ Command-line interface for MONAI Histo Segmenter: a simple showcase for histopathological image 
    segmentation with MONAI """

    # Setup ArgumentParser interface
    parser = ArgumentParser(prog="MONAI Histo Segmenter", description=desc)

    # Add optional core arguments
    parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s_v" + VERSION
    )

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


def __cli_train(subparsers):
    """ """
    desc = """ Pipeline hub for Training the MONAI model on a tiled dataset """

    parser_train = subparsers.add_parser("train", help=desc, add_help=False)

    setup_path_args(parser_train, "train")
    setup_monai_args(parser_train, "train")
    __setup__help_args(parser_train)


def __cli_test(subparsers):
    """ """
    desc = """ Pipeline hub for Testing a model trained with the training hub """

    parser_train = subparsers.add_parser("test", help=desc, add_help=False)

    setup_path_args(parser_train, "test")
    setup_monai_args(parser_train, "test")
    __setup__help_args(parser_train)


def __cli_onnx_export(subparsers):
    """ """
    desc = """ Pipeline hub for Exporting a model trained with the training hub """

    parser_train = subparsers.add_parser("export", help=desc, add_help=False)

    setup_path_args(parser_train, "export")
    setup_monai_args(parser_train, "export")
    __setup__help_args(parser_train)


def __setup__help_args(parser: ArgumentParser) -> None:
    arg_group = parser.add_argument_group("Arguments - Other")
    arg_group.add_argument(
        "-h", "--help", action="help", help="show this help message and exit"
    )
