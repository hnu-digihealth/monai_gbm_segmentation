from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.onnx
from monai.data import DataLoader, Dataset
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensor, MapTransform


# Define the HE normalization function
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
            img = d[key].permute(1, 2, 0).cpu().numpy()
            img = normalize_staining(img, self.target_mean, self.target_std)
            d[key] = torch.from_numpy(img).permute(2, 0, 1)
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
            img = d[key]
            img_single_channel = np.zeros_like(img[0], dtype=np.uint8)
            img_single_channel[img[0] == 0] = 0
            img_single_channel[img[0] == 255] = 1
            d[key] = torch.tensor(img_single_channel, dtype=torch.float32).unsqueeze(0)
        return d

def main():
    # Define the device variable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the model with the same parameters as during training
    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=0.1
    )

    # Load the saved model weights
    base_dir = Path(__file__).resolve().parent.parent

    model_path = base_dir / 'Modelle' / 'model1.pth'
    if not model_path.is_file():
        print(f"Model path does not exist: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))

    # Set the model to evaluation mode
    model.to(device)
    model.eval()

    # Export the model to ONNX format
    onnx_model_path = base_dir / 'Modelle' / 'model_1.onnx'
    dummy_input = torch.randn(1, 3, 1024, 1024).to(device) # Example input size (batch size, channels, height, width)
    torch.onnx.export(
        model,                       # The model to be exported
        dummy_input,                 # Example input data
        str(onnx_model_path),        # Name of the output ONNX file
        export_params=True,          # Store the trained parameter weights inside the model file
        opset_version=11,            # The ONNX version to export the model to
        do_constant_folding=True,    # Perform constant folding for optimization
        input_names=['input'],       # Input names
        output_names=['output'],     # Output names
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Dynamic axes for batch size
    )

    print("Model successfully exported to ONNX.")

    # Verification process
    def verify_model(pytorch_model, onnx_model_path, input_tensor):
        """
        Verify the ONNX model against the PyTorch model.

        Args:
            pytorch_model (torch.nn.Module): The PyTorch model.
            onnx_model_path (str): Path to the ONNX model file.
            input_tensor (torch.Tensor): Input tensor for the models.

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
            np.testing.assert_allclose(pytorch_output.cpu().numpy(), onnx_output.numpy(), rtol=1e-02, atol=1e-03)
            return True, None
        except AssertionError as e:
            diff = np.abs(pytorch_output.cpu().numpy() - onnx_output.numpy())
            max_abs_diff = np.max(diff)
            max_rel_diff = np.max(diff / (np.abs(onnx_output.numpy()) + 1e-8))
            return False, (max_abs_diff, max_rel_diff)

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

    # Create test data
    test_files = [{'img': str(img), 'seg': str(seg)} for img, seg in zip(test_images, test_labels)]

    # Transformation pipelines
    test_transforms = Compose([
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        HENormalization(keys=["img"], target_mean=(0.485, 0.456, 0.406), target_std=(0.229, 0.224, 0.225)),
        ConvertToSingleChannel(keys=["seg"]),
        ScaleIntensityd(keys=["img"]),
        ToTensor()
    ])

    # Create dataset and dataloader
    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_data_loader = DataLoader(test_ds, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())

    results = []
    for batch in test_data_loader:
        input_tensor = batch['img'].to(device)
        result, diffs = verify_model(model,  str(onnx_model_path), input_tensor)
        results.append((result, diffs))

    # Output cumulative results
    successful = sum(1 for result, _ in results if result)
    failed = len(results) - successful
    print(f"Number of successful verifications: {successful}")
    print(f"Number of failed verifications: {failed}")
    if failed > 0:
        for i, (result, diffs) in enumerate(results):
            if not result:
                print(f"Failed verification {i+1}: Max absolute difference: {diffs[0]}, Max relative difference: {diffs[1]}")

if __name__ == '__main__':
    main()
