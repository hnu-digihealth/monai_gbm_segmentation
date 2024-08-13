from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.onnx
from monai.data import DataLoader, Dataset
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, MapTransform


def normalize_staining(image: np.ndarray, target_mean=(0.485, 0.456, 0.406), target_std=(0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Normalize the staining of the image.

    Args:
        image (np.ndarray): Input image.
        target_mean (tuple): Target mean values for normalization.
        target_std (tuple): Target standard deviation values for normalization.

    Returns:
        np.ndarray: Normalized image.
    """
    image = image.astype(np.float32) / 255.0
    means, stds = cv2.meanStdDev(image)
    means = means.reshape(3)
    stds = stds.reshape(3)
    norm_image = np.zeros_like(image)
    for i in range(3):
        norm_image[:, :, i] = (image[:, :, i] - means[i]) / stds[i] * target_std[i] + target_mean[i]
    norm_image = np.clip(norm_image, 0, 1)
    norm_image = (norm_image * 255).astype(np.uint8)
    return norm_image

class HENormalization(MapTransform):
    """
    Apply H&E normalization to images.
    """
    def __init__(self, keys, target_mean=(0.485, 0.456, 0.406), target_std=(0.229, 0.224, 0.225)):
        super().__init__(keys)
        self.target_mean = target_mean
        self.target_std = target_std

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key].permute(1, 2, 0).cpu().numpy()  # Convert from CHW to HWC for OpenCV and to numpy
            img = normalize_staining(img, self.target_mean, self.target_std)
            d[key] = torch.from_numpy(img).permute(2, 0, 1)  # Convert back from HWC to CHW
        return d

class ConvertToSingleChannel(MapTransform):
    """
    Convert the image to a single channel.
    """
    def __init__(self, keys):
        super().__init__(keys)
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key].permute(1, 2, 0).cpu().numpy()
            img_single_channel = np.zeros_like(img[..., 0], dtype=np.uint8)  # Use only the first channel
            img_single_channel[img[..., 0] == 0] = 0  # Tumor (black)
            img_single_channel[img[..., 0] == 255] = 1  # Healthy (white)
            d[key] = torch.tensor(img_single_channel, dtype=torch.float32).unsqueeze(0)
        return d

def main():
    # Define the device variable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the model with the same parameters as during training
    model = UNet(
        spatial_dims=2,
        in_channels=3,  # 3 for RGB images
        out_channels=1,  # 1 for binary segmentation
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,  # Number of residual units in each layer
        norm=Norm.BATCH,  # Batch normalization
        dropout=0.1
    )

    # Load the saved model weights
    base_dir = Path(__file__).resolve().parent.parent

    model_path = base_dir / 'Modelle' / 'model2.ckpt'
    if not model_path.is_file():
        print(f"Model path does not exist: {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict']
    new_state_dict = {k[len('model.'):]: v for k, v in state_dict.items() if k.startswith('model.')}
    model.load_state_dict(new_state_dict)

    # Set the model to evaluation mode
    model.to(device)
    model.eval()

    # Export the model to ONNX format
    onnx_model_path = base_dir / 'Modelle' / 'model_2.onnx'
    dummy_input = torch.randn(1, 3, 1024, 1024).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_model_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print("Model successfully exported to ONNX.")

    # Verification process
    def verify_model(pytorch_model, onnx_model_path, input_tensor, rtol=1e-02, atol=1e-03):
        """
        Verify the ONNX model against the PyTorch model.

        Args:
            pytorch_model (torch.nn.Module): The PyTorch model.
            onnx_model_path (str): Path to the ONNX model file.
            input_tensor (torch.Tensor): Input tensor for the models.
            rtol (float): Relative tolerance for comparison.
            atol (float): Absolute tolerance for comparison.

        Returns:
            tuple: (bool, tuple) indicating if the verification was successful and the differences if not.
        """
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(input_tensor)
        
        ort_session = ort.InferenceSession(onnx_model_path)
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        onnx_output = torch.tensor(ort_outs[0])

        try:
            np.testing.assert_allclose(pytorch_output.cpu().detach().numpy(), onnx_output.numpy(), rtol=rtol, atol=atol)
            return True, None
        except AssertionError as e:
            diff = np.abs(pytorch_output.cpu().detach().numpy() - onnx_output.numpy())
            max_abs_diff = np.max(diff)
            max_rel_diff = np.max(diff / (np.abs(onnx_output.numpy()) + 1e-8))
            return False, (max_abs_diff, max_rel_diff)

    def compare_intermediate_layers(pytorch_model, onnx_model_path, input_tensor, rtol=1e-02, atol=1e-03):
        """
        Compare the intermediate layer outputs of the PyTorch and ONNX models.

        Args:
            pytorch_model (torch.nn.Module): The PyTorch model.
            onnx_model_path (str): Path to the ONNX model file.
            input_tensor (torch.Tensor): Input tensor for the models.
            rtol (float): Relative tolerance for comparison.
            atol (float): Absolute tolerance for comparison.
        """
        intermediate_outputs_pytorch = []
        intermediate_outputs_onnx = []

        def hook(module, input, output):
            intermediate_outputs_pytorch.append(output)

        hooks = []
        for layer in pytorch_model.modules():
            if isinstance(layer, torch.nn.Conv2d):  # Example layer type
                hooks.append(layer.register_forward_hook(hook))

        # Forward pass through PyTorch model
        pytorch_model(input_tensor)

        # Remove hooks
        for h in hooks:
            h.remove()

        # ONNX prediction
        ort_session = ort.InferenceSession(onnx_model_path)
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.cpu().numpy()}
        intermediate_outputs_onnx = ort_session.run(None, ort_inputs)

        for i, (pyt_output, onnx_output) in enumerate(zip(intermediate_outputs_pytorch, intermediate_outputs_onnx)):
            try:
                np.testing.assert_allclose(pyt_output.cpu().detach().numpy(), onnx_output, rtol=rtol, atol=atol)
                print(f"Layer {i} matches")
            except AssertionError:
                print(f"Layer {i} does not match")
                print(f"PyTorch output: {pyt_output.cpu().detach().numpy()}")
                print(f"ONNX output: {onnx_output}")

    test_images_path = base_dir.parent / 'Data' / 'test_data_t3' / 'test3' / 'img'
    test_labels_path = base_dir.parent / 'Data' / 'test_data_t3' / 'test3' / 'lbl'

    # Check if the test data paths exist
    if not test_images_path.is_dir():
        print(f"Test images path does not exist: {test_images_path}")
        return

    if not test_labels_path.is_dir():
        print(f"Test labels path does not exist: {test_labels_path}")
        return

    test_images = sorted([x for x in test_images_path.iterdir() if x.suffix == '.png'])
    test_labels = sorted([x for x in test_labels_path.iterdir() if x.suffix == '.png'])
    test_files = [{'img': img, 'seg': seg} for img, seg in zip(test_images, test_labels)]

    test_transforms = Compose([
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        HENormalization(keys=["img"], target_mean=(0.485, 0.456, 0.406), target_std=(0.229, 0.224, 0.225)),
        ConvertToSingleChannel(keys=["seg"]),
        ScaleIntensityd(keys=["img"]),
        ToTensord(keys=["img", "seg"])
    ])

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_data_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    successful_verifications = 0
    failed_verifications = 0
    failed_verifications_details = []

    # Initial Tolerances
    rtol = 1e-02
    atol = 1e-03

    for i, test_data in enumerate(test_data_loader):
        input_tensor = test_data["img"].to(device)
        is_successful, diff_details = verify_model(model, str(onnx_model_path), input_tensor, rtol, atol)
        if is_successful:
            successful_verifications += 1
        else:
            failed_verifications += 1
            failed_verifications_details.append((i, diff_details))

    print(f"Number of successful verifications: {successful_verifications}")
    print(f"Number of failed verifications: {failed_verifications}")

    for idx, (max_abs_diff, max_rel_diff) in failed_verifications_details:
        print(f"Failed verification {idx}: Max absolute difference: {max_abs_diff}, Max relative difference: {max_rel_diff}")

    # Compare the intermediate layer outputs
    input_tensor = next(iter(test_data_loader))["img"].to(device)
    compare_intermediate_layers(model, str(onnx_model_path), input_tensor, rtol, atol)

if __name__ == '__main__':
    main()
