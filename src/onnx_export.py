"""
ONNX export module for MONAI-based UNet segmentation model.

Loads a trained model checkpoint and exports it to ONNX format
for interoperability with deployment frameworks.
"""

# Python Standard Library
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Third Party Libraries
import torch
import torchstain
from monai.data import Dataset
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, ToTensor
from monai.utils.enums import MetaKeys, TraceKeys
from torch.serialization import add_safe_globals
from torchvision.io import read_image

# Local Libraries
from src.helper_functions.machine_learning import init_unet_model
from src.helper_functions.preprocessing import HENormalization

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


def prepare_model(model_path: Path) -> torch.nn.Module:
    """
    Initialize and prepare the MONAI UNet model for ONNX export.

    This function instantiates the model architecture, loads trained weights from
    the checkpoint, moves the model to the appropriate device (CPU/GPU), and sets
    it to evaluation mode.

    Args:
        model_path (Path): Path to the .ckpt model checkpoint.

    Returns:
        torch.nn.Module: A fully prepared and initialized model ready for export or inference.
    """
    model = init_unet_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    add_safe_globals([MetaTensor, MetaKeys, TraceKeys])
    state_dict = load_checkpoint_weights(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def verify_model(pytorch_model, onnx_model_path, input_tensor, rtol=1e-02, atol=1e-03, return_diff=False):
    """
    Verify the ONNX model against the PyTorch model using a given input tensor.
    Logs detailed differences if verification fails.

    Args:
        pytorch_model: Trained PyTorch model (e.g. MONAI UNet).
        onnx_model_path: Path to the exported ONNX file.
        input_tensor: Input tensor for inference (e.g. real or dummy image).
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
        return_diff (bool): Whether to return diff statistics for further analysis.

    Returns:
        None or dict with diff values if return_diff is True
    """
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(input_tensor)

    ort_session = ort.InferenceSession(str(onnx_model_path))
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_output = torch.tensor(ort_outs[0])

    try:
        np.testing.assert_allclose(pytorch_output.cpu().detach().numpy(), onnx_output.numpy(), rtol=rtol, atol=atol)
        logger.info("ONNX model verification successful – outputs are within tolerance.")
        return None if return_diff else None
    except AssertionError:
        diff = np.abs(pytorch_output.cpu().detach().numpy() - onnx_output.numpy())
        max_abs_diff = np.max(diff)
        mean_abs_diff = np.mean(diff)
        max_rel_diff = np.max(diff / (np.abs(onnx_output.numpy()) + 1e-8))

        logger.warning("ONNX model verification failed.")
        logger.warning(f"Max absolute difference: {max_abs_diff:.6f}")
        logger.warning(f"Mean absolute difference: {mean_abs_diff:.6f}")
        logger.warning(f"Max relative difference: {max_rel_diff:.6f}")

        if return_diff:
            return {
                "max_abs_diff": float(max_abs_diff),
                "mean_abs_diff": float(mean_abs_diff),
                "max_rel_diff": float(max_rel_diff),
            }


def export_model(
    model_path: Path,
    mode: str,
    test_image_path: Path | None = None,
    normalizer_image_path: Path | None = None,
) -> None:
    """
    Export a trained MONAI-based UNet model to ONNX format and verify its output.
    Logs all progress and differences to the log output.  If available, also logs error statistics.
    This function performs the following steps:
    1. Loads and prepares the model from a .ckpt checkpoint.
    2. Exports the model to ONNX format using a dummy input.
    3. Verifies ONNX model consistency using dummy and (optionally) real image input.
       If a folder of images is given, it will batch-verify each one individually.

    Args:
        model_path (Path): Path to the trained PyTorch Lightning checkpoint (.ckpt).
        mode (str): Precision mode (currently unused, reserved for future options).
        test_image_path (Path | None): Either a single test image or a directory of PNG images.
        normalizer_image_path (Path | None): Path to the H&E reference image for stain normalization.

    Returns:
        None.
    """
    logger.info("Starting ONNX export process")
    logger.info(f"Loading model from checkpoint: {model_path}")

    model = prepare_model(model_path)

    # Define export path relative to checkpoint
    device = next(model.parameters()).device

    # Dummy input
    dummy_input = torch.randn(1, 3, 1024, 1024, device=device)
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

    # Dummy verification
    logger.info("Verifying ONNX model using dummy input...")
    verify_model(model, export_path, dummy_input)

    # Optional: real image verification
    if test_image_path and test_image_path.exists() and normalizer_image_path and normalizer_image_path.exists():
        try:
            logger.info(f"Initializing stain normalizer from: {normalizer_image_path}")
            normalizer = torchstain.normalizers.ReinhardNormalizer(method="modified", backend="torch")
            normalizer.fit(read_image(normalizer_image_path))

            test_images = [test_image_path] if test_image_path.is_file() else sorted(test_image_path.glob("*.png"))
            logger.info(f"Found {len(test_images)} image(s) for ONNX verification")

            diffs = []

            for img_path in test_images:
                logger.info(f"Processing image: {img_path.name}")
                test_data = [{"img": img_path}]
                transforms = Compose(
                    [
                        LoadImaged(keys=["img"], dtype=np.float32, ensure_channel_first=True),
                        HENormalization(keys=["img"], normalizer=normalizer, method="reinhard"),
                        EnsureChannelFirstd(keys=["img"]),
                        ToTensor(dtype=np.float32),
                    ]
                )
                dataset = Dataset(data=test_data, transform=transforms)
                sample = dataset[0]
                input_tensor = sample["img"].unsqueeze(0).to(device)
                diff = verify_model(model, export_path, input_tensor, return_diff=True)
                if diff:
                    diffs.append(diff)

            if diffs:
                max_diffs = [d["max_abs_diff"] for d in diffs]
                mean_diffs = [d["mean_abs_diff"] for d in diffs]
                logger.info("ONNX verification summary over real images:")
                logger.info(f"- Average max absolute difference: {np.mean(max_diffs):.6f}")
                logger.info(f"- Average mean absolute difference: {np.mean(mean_diffs):.6f}")
        except Exception as e:
            logger.warning(f"Real image ONNX verification failed: {e}")
    else:
        logger.info("No real test image or normalizer provided – skipping real image verification.")
