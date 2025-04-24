# Python Standard Library
from argparse import Namespace

# Third Party Libraries
from monai.utils.misc import set_determinism

# Local Libraries
from src.argparser.argparser_setup import cli_core
from src.onnx_export import export_model
from src.testing import test_model
from src.training import train_and_validate_model
from torch import set_float32_matmul_precision


def main(args: Namespace):
    # set MONAI seed for all following random operations
    set_determinism(seed=args.seed)
    # set pytorch float32 matmul precision
    set_float32_matmul_precision("medium")

    match args.hub:
        case "train":
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
            test_model(
                args.test_path,
                args.normalizer_image_path,
                args.batch_size,
                args.num_workers,
                args.model_path,
            )
        case "export":
            export_model(args.model_path, args.mode, args.test_path)


if __name__ == "__main__":
    parser = cli_core()
    args = parser.parse_args()
    print(args)
    main(args)
