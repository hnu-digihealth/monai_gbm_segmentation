"""
ONNX export module for MONAI-based UNet segmentation model.

Loads a trained model checkpoint and exports it to ONNX format
for interoperability with deployment frameworks.
"""

# Python Standard Library
import logging
from pathlib import Path

# Third Party Libraries
import torch

# Local Libraries
from src.helper_functions.machine_learning import init_unet_model

# Setup logger
logger = logging.getLogger("ONNXExport")


# Load weights from PyTorch Lightning checkpoint and strip prefix if needed
def load_checkpoint_weights(model_path: Path) -> dict:
    """
    Loads and optionally strips state dict from checkpoint.

    Args:
        model_path (Path): Path to .ckpt checkpoint file.

    Returns:
        dict: Processed state_dict for model loading.
    """
    checkpoint = torch.load(model_path)
    state_dict = checkpoint["state_dict"]
    return {k[len("model.") :] if k.startswith("model.") else k: v for k, v in state_dict.items()}


# Initialize and prepare model for ONNX export (weights, device, eval mode)
def prepare_model(model_path: Path) -> torch.nn.Module:
    """
    Initializes model, loads weights, moves to device and sets eval mode.

    Args:
        model_path (Path): Path to .ckpt file.

    Returns:
        torch.nn.Module: Loaded model ready for export.
    """
    # Create model instance with architecture identical to training
    model = init_unet_model()
    # Select device based on availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    state_dict = load_checkpoint_weights(model_path)
    # Load weights into the model
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def export_model(
    model_path: Path,
    mode: str,
    test_image_path: Path | None = None,
) -> None:
    """
    Export a trained PyTorch model to ONNX format.

    Args:
        model_path (Path): Path to the model checkpoint (.ckpt).
        mode (str): Precision mode (unused but kept for interface compatibility).
        test_image_path (Path | None): Optional test image path (unused).

    Returns:
        None
    """
    logger.info("Starting ONNX export process")
    logger.info(f"Loading model from checkpoint: {model_path}")

    model = prepare_model(model_path)

    # Dummy input simulating a 1024x1024 RGB image (1 batch)
    dummy_input = torch.randn(1, 3, 1024, 1024, device=next(model.parameters()).device)
    # Define export path relative to checkpoint
    export_path = model_path.parent / "model.onnx"

    logger.info(f"Exporting model to ONNX at {export_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(export_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    # Finished export
    logger.info("ONNX export completed successfully")
