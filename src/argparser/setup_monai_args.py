"""
Argument group setup for MONAI-related CLI arguments.

Adds arguments for training mode, device selection, batch size, and worker threads.
"""

# Python Standard Library
from argparse import ArgumentParser


def setup_monai_args(parser: ArgumentParser, hub_type: str) -> None:
    """
    Adds the `Arguments - MONAI Setup` group to the provided parser by reference. The required arguments are added based
    on the provided `hub_type` argument.

    Args:
        parser (ArgumentParser): The CLI parser to extend.
        hub_type (str): Application mode ('train', 'test', 'export') to determine which arguments are relevant.

    Returns:
        None
    """
    # add MONAI argument group
    arg_group = parser.add_argument_group("Arguments - MONAI Setup")

    arg_group.add_argument(
        "-m",
        "--mode",
        required=True,
        default="gpu",
        choices=["gpu", "cpu"],
        help="If the ML should use `gpu` or `cpu` mode.",
    )

    arg_group.add_argument(
        "-d",
        "--devices",
        required=False,
        default=None,
        help="The GPU devices to use. Only relevant if mode is `gpu`. By default all available devices are used.",
    )

    if hub_type != "export":
        arg_group.add_argument(
            "-w",
            "--num_workers",
            required=False,
            default=4,
            type=int,
            help="Number of workers to use for data loading.",
        )

        arg_group.add_argument(
            "-b",
            "--batch_size",
            required=False,
            default=32,
            type=int,
            help="Batch size to use for training/testing.",
        )

        arg_group.add_argument(
            "-s",
            "--seed",
            type=int,
            required=False,
            default="421337133742",
            help="Seed used by MONAI for random operations.",
        )
