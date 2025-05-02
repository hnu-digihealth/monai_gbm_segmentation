import shutil
from argparse import Namespace
from pathlib import Path

import torch
from monai_segmenter import main
from src.logging.setup_logger import setup_logger

if __name__ == "__main__":

    gpu_available = torch.cuda.is_available()
    devices = "0" if gpu_available else "cpu"
    src_tile = Path(r"C:\Users\Ellena\Documents\minimal_opt\split_dataset\train_tiles\images")
    normalizer_target = Path(r"C:\Users\Ellena\Documents\minimal_opt\split_dataset\normalizer.png")

    if not normalizer_target.exists():
        example_img = next(src_tile.glob("*.*"))
        shutil.copy(example_img, normalizer_target)
        print(f"Create normalizer: {normalizer_target}")

    args = Namespace(
        hub="train",
        train_path=Path("C:/Users/Ellena/Documents/minimal_opt/split_dataset/train_tiles"),
        val_path=Path("C:/Users/Ellena/Documents/minimal_opt/split_dataset/val_tiles"),
        normalizer_image_path=Path("C:/Users/Ellena/Documents/minimal_opt/split_dataset/normalizer.png"),
        batch_size=4,
        mode="auto",
        devices=devices,
        model_path=Path("C:/Users/Ellena/Documents/output/demo_model.pth"),
        num_workers=2,
        seed=42,
        log_level="DEBUG",
    )

    logger = setup_logger("Segmenter", level=args.log_level)
    main(args, logger)
