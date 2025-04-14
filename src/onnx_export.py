# Python Standard Library
from pathlib import Path

# Third Party Libraries
import torch

# Local Libraries
from src.helper_functions.machine_learning import init_unet_model


def export_model(
    model_path: Path,
    mode: str,
    test_image_path: Path | None = None,
) -> None:
    model = init_unet_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {
        k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")
    }
    model.load_state_dict(new_state_dict)

    # Set the model to evaluation mode
    model.to(device)
    model.eval()

    # Export the model to ONNX format
    dummy_input = torch.randn(1, 3, 1024, 1024).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        str(model_path.parent / "model.onnx"),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
