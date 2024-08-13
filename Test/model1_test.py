from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from tqdm import tqdm
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensor, MapTransform, Resized
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric

# Define the HE normalization function
def normalize_staining(image, target_mean=(0.485, 0.456, 0.406), target_std=(0.229, 0.224, 0.225)):
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
            img = d[key]
            img_single_channel = np.zeros_like(img[0], dtype=np.uint8)  # Use only the first channel
            img_single_channel[img[0] == 0] = 0  # Tumor (black)
            img_single_channel[img[0] == 255] = 1  # Healthy (white)
            d[key] = torch.tensor(img_single_channel, dtype=torch.float32).unsqueeze(0)
        return d

def visualize_predictions(model, data_loader, device, num_samples=5):
    """
    Visualize model predictions.

    Args:
        model (torch.nn.Module): The model.
        data_loader (DataLoader): DataLoader for the data.
        device (torch.device): Device to use for computation.
        num_samples (int): Number of samples to visualize.
    """
    model.eval()
    with torch.no_grad():
        count = 0
        for batch in data_loader:
            inputs = batch['img'].to(device)
            labels = batch['seg'].to(device)
            
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5

            for j in range(inputs.shape[0]):
                if count >= num_samples:
                    return
                plt.figure(figsize=(18, 6))
                
                plt.subplot(1, 3, 1)
                plt.title("Original Image")
                img = inputs[j].cpu().permute(1, 2, 0).numpy()
                plt.imshow(img)
                
                plt.subplot(1, 3, 2)
                plt.title("Ground Truth Mask")
                gt_mask = labels[j][0].cpu().numpy()
                plt.imshow(gt_mask, cmap='gray')
                
                plt.subplot(1, 3, 3)
                plt.title("Predicted Mask")
                pred_mask = preds[j][0].cpu().numpy()
                plt.imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
                
                plt.show()
                count += 1

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path(__file__).resolve().parent.parent
    test_images_path = base_dir.parent / 'Data' / 'test_data_t3' / 'test3' / 'img'
    test_labels_path = base_dir.parent / 'Data' / 'test_data_t3' / 'test3' / 'lbl'


    test_images = sorted([x for x in test_images_path.iterdir() if x.suffix == '.png'])
    test_labels = sorted([x for x in test_labels_path.iterdir() if x.suffix == '.png'])

    test_files = [{'img': img, 'seg': seg} for img, seg in zip(test_images, test_labels)]

    test_transforms = Compose([
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        HENormalization(keys=["img"], target_mean=(0.485, 0.456, 0.406), target_std=(0.229, 0.224, 0.225)),
        ConvertToSingleChannel(keys=["seg"]),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img", "seg"], spatial_size=(1024, 1024)),
        ToTensor()
    ])

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=16, num_workers=8, pin_memory=torch.cuda.is_available())

    model_path = base_dir / 'Modelle' / 'model1.pth'
    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=0.1
    ).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    iou_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
    cm_metric = ConfusionMatrixMetric(metric_name=["precision", "recall", "f1_score", "accuracy"], include_background=True, reduction="mean", get_not_nans=False)

    with torch.no_grad():
        dice_metric.reset()
        iou_metric.reset()
        cm_metric.reset()
        for test_data in tqdm(test_loader, desc="Testing model"):
            test_inputs, test_labels = test_data["img"].to(device), test_data["seg"].to(device)
            test_outputs = model(test_inputs)
            test_outputs = torch.sigmoid(test_outputs)
            test_outputs = (test_outputs > 0.5).float()

            dice_metric(y_pred=test_outputs, y=test_labels)
            iou_metric(y_pred=test_outputs, y=test_labels)
            cm_metric(y_pred=test_outputs, y=test_labels)
        
        dice_value = dice_metric.aggregate().item()
        mean_iou = iou_metric.aggregate().item()
        precision, recall, f1_score, accuracy = cm_metric.aggregate()
        
        mean_precision = precision.item()
        mean_recall = recall.item()
        mean_f1 = f1_score.item()
        mean_accuracy = accuracy.item()

        dice_metric.reset()
        iou_metric.reset()
        cm_metric.reset()
        
        print(f"Test Dice: {dice_value:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"Mean Precision: {mean_precision:.4f}")
        print(f"Mean Recall: {mean_recall:.4f}")
        print(f"Mean F1-Score: {mean_f1:.4f}")
        print(f"Mean Accuracy: {mean_accuracy:.4f}")

    visualize_predictions(model, test_loader, device, num_samples=5)
