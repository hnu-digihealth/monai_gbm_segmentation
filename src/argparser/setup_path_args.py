# Python Standard Library
from argparse import ArgumentParser


def setup_path_args(parser: ArgumentParser, hub_type: str) -> None:
    """Adds the `Arguments - Path Setup` group to the provided parser by reference. The required paths are added based on
    the provided `hub_type` argument.

    Parameters
    ----------
    - `parser: ArgumentParser` - The ArgumentParser to add the argument group to
    - `hub_type: str`- Hub type to determine the required arguments to add to the group. Should be one of
        `train`, `test`, and `export`.

    Returns
    -------
    - `None`

    """
    # add IO argument group
    arg_group = parser.add_argument_group("Arguments - Path Setup")

    # model path and normalizer image path are always required
    arg_group.add_argument(
        "-mp",
        "--model_path",
        required=True,
        help=(
            "Path of the trained model to export to ONNX. Resulting ONNX model will be saved at same path as ONNX."
            if hub_type == "export"
            else (
                "Path to where the model should be saved to."
                if hub_type == "train"
                else "Path to the trained model to test."
            )
        ),
    )
    arg_group.add_argument(
        "-nip",
        "--normalizer_image_path",
        required=True,
        help="Path to the normalizer image. This image will be used to normalize the input images.",
    )

    # train needs train images
    if hub_type == "train":
        arg_group.add_argument(
            "-tp",
            "--train_path",
            required=True,
            help="Path to the train images folder. The folder should contain a img and lbl subfolder.",
        )
        arg_group.add_argument(
            "-vp",
            "--val_path",
            required=True,
            help="Path to the validation images folder. The folder should contain a img and lbl subfolder.",
        )

    # testing and export need test folder
    if hub_type != "train":
        arg_group.add_argument(
            "-tp",
            "--test_path",
            required=True if hub_type == "test" else False,
            help="Path to the test images folder. The folder should contain a img and lbl subfolder.",
        )
