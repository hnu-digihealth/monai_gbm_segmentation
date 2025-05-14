from pathlib import Path

from src.onnx_export import export_model


def main():
    # Path to the model checkpoint (.ckpt)
    model_ckpt_path = Path(
        r"C:\Users\Ellena\Documents\output\demo_model.pth\lightning_logs\version_19\checkpoints\epoch=3-step=80.ckpt"
    )

    # Path to the test image folder (contains multiple .png images)
    test_image_path = Path(r"C:\Users\Ellena\Documents\minimal_opt\split_dataset\train_tiles\img")

    # Path to the H&E normalization reference image
    normalizer_image_path = Path(r"C:\Users\Ellena\Documents\minimal_opt\split_dataset\normalizer.png")

    # Validate all required files/folders exist
    if not model_ckpt_path.is_file():
        print(f"❌ Model checkpoint not found: {model_ckpt_path}")
        return
    if not test_image_path.exists():
        print(f"❌ Test image (file or folder) not found: {test_image_path}")
        return
    if not normalizer_image_path.is_file():
        print(f"❌ Normalizer reference image not found: {normalizer_image_path}")
        return

    try:
        print("🚀 Starting ONNX export and validation with real images...")
        export_model(
            model_path=model_ckpt_path,
            mode="default",
            test_image_path=test_image_path,
            normalizer_image_path=normalizer_image_path,
        )
        print("✅ ONNX export and multi-image validation completed successfully.")
    except Exception as e:
        print(f"❌ Export or validation failed: {e}")


if __name__ == "__main__":
    main()
