from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import time
import traceback
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensor, RandRotate90d, 
    RandFlipd, MapTransform, RandAdjustContrastd, RandGaussianNoised, 
    RandZoomd, RandAffined
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, FocalLoss
from monai.metrics import DiceMetric
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Define the device variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Debugging function to check transformations
def debug_transformations(dataset, original_files, num_samples=5):
    for i in range(num_samples):
        try:
            sample = dataset[i]
            original_image_path = original_files[i]['img']
            original_image = cv2.imread(str(original_image_path))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            original_segmentation_path = original_files[i]['seg']
            original_segmentation = cv2.imread(str(original_segmentation_path), cv2.IMREAD_GRAYSCALE)
            
            print(f"Sample {i} - Image shape: {sample['img'].shape}, Seg shape: {sample['seg'].shape}")
            print(f"Original segmentation unique values: {np.unique(original_segmentation)}")
            print(f"Transformed segmentation unique values: {np.unique(sample['seg'].cpu().numpy())}")
            
            plt.figure(figsize=(18, 6))
            
            plt.subplot(1, 4, 1)
            plt.title("Original Image")
            plt.imshow(original_image)  # Display original image
            
            plt.subplot(1, 4, 2)
            plt.title("Transformed Image")
            transformed_img = sample['img'].permute(1, 2, 0).cpu().numpy()
            plt.imshow(transformed_img)  # Display transformed image
            
            plt.subplot(1, 4, 3)
            plt.title("Original Segmentation")
            plt.imshow(original_segmentation, cmap='gray')
            
            plt.subplot(1, 4, 4)
            plt.title("Transformed Segmentation")
            plt.imshow(sample['seg'][0].cpu().numpy(), cmap='gray')
            
            plt.show()
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            traceback.print_exc()

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

class CombinedDiceFocalLoss(nn.Module):
    """
    Combined Dice and Focal loss.
    """
    def __init__(self, dice_weight=0.7, focal_weight=0.3):
        super(CombinedDiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
        self.focal_loss = FocalLoss()

    def forward(self, outputs, targets):
        dice_loss = self.dice_loss(outputs, targets)
        focal_loss = self.focal_loss(outputs, targets)
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss    

# Define paths for training and validation data
# Define the base directory relative to the script location
base_dir = Path(__file__).resolve().parent.parent

# Define paths for training and validation data
data_path = base_dir.parent / 'Data' / 'training_data'

train_images_path = data_path / 'train' / 'img'
train_labels_path = data_path / 'train' / 'lbl'
val_images_path = data_path / 'validate' / 'img'
val_labels_path = data_path / 'validate' / 'lbl'

train_images = sorted([x for x in train_images_path.iterdir() if x.suffix == '.png'])
train_labels = sorted([x for x in train_labels_path.iterdir() if x.suffix == '.png'])
val_images = sorted([x for x in val_images_path.iterdir() if x.suffix == '.png'])
val_labels = sorted([x for x in val_labels_path.iterdir() if x.suffix == '.png'])

# Create training and validation data
train_files = [{'img': img, 'seg': seg} for img, seg in zip(train_images, train_labels)]
val_files = [{'img': img, 'seg': seg} for img, seg in zip(val_images, val_labels)]

# Transformation pipelines
train_transforms = Compose([
    LoadImaged(keys=["img", "seg"]),
    EnsureChannelFirstd(keys=["img", "seg"]),
    HENormalization(keys=["img"], target_mean=(0.485, 0.456, 0.406), target_std=(0.229, 0.224, 0.225)),
    ConvertToSingleChannel(keys=["seg"]),
    ScaleIntensityd(keys=["img"]),
    RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["img", "seg"], prob=0.5),
    RandAdjustContrastd(keys=["img"], prob=0.5, gamma=(0.7, 1.3)),
    RandGaussianNoised(keys=["img"], prob=0.5, mean=0.0, std=0.01),
    RandZoomd(keys=["img", "seg"], prob=0.5, min_zoom=0.9, max_zoom=1.1),
    RandAffined(keys=["img", "seg"], prob=0.5, rotate_range=(0.1, 0.1), shear_range=(0.1, 0.1)),
    ToTensor()
])

val_transforms = Compose([
    LoadImaged(keys=["img", "seg"]),
    EnsureChannelFirstd(keys=["img", "seg"]),
    HENormalization(keys=["img"], target_mean=(0.485, 0.456, 0.406), target_std=(0.229, 0.224, 0.225)),
    ConvertToSingleChannel(keys=["seg"]),
    ScaleIntensityd(keys=["img"]),
    ToTensor()
])

# Create datasets and dataloaders
train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)

if __name__ == "__main__":
    # Debugging the transformation pipeline
    debug_transformations(train_ds, train_files, num_samples=1)
    val_loader = DataLoader(val_ds, batch_size=16, num_workers=8, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=16, num_workers=8, shuffle=True, pin_memory=torch.cuda.is_available())

    # Define model, loss, and metric
    model = UNet(
        spatial_dims=2,
        in_channels=3,  # 3 for RGB images
        out_channels=1,  # 1 for binary segmentation
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,  # Number of residual units in each layer
        norm=Norm.BATCH,  # Batch normalization
        dropout=0.1
    ).to(device)

    # Use DataParallel to use the model on multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Define loss and metric functions
    loss_function = CombinedDiceFocalLoss(dice_weight=0.7, focal_weight=0.3)
    metric = DiceMetric(include_background=True, reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Training and validation
    max_epochs = 100
    best_metric = -1
    best_metric_epoch = -1
    patience = 10  # Number of epochs with no improvement after which training will be stopped
    counter = 0

    for epoch in range(max_epochs):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{max_epochs}, Current learning rate: {current_lr}")
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{max_epochs}") as pbar:  # Initialize progress bar
            for batch_data in train_loader:
                inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()

                pbar.update(1)  # Update progress bar

        epoch_loss /= len(train_loader)
        print(f"Average Training Loss: {epoch_loss:.4f}")
        code_time = time.time() - start_time
        print(f"Time for Epoch: {code_time:.2f} seconds")

        # Validation
        model_path = base_dir / 'Modelle' / 'model3.pth'
        model.eval()
        with torch.no_grad():
            metric.reset()
            for val_data in val_loader:
                val_inputs, val_labels = val_data["img"].to(device), val_data["seg"].to(device)

                val_outputs = model(val_inputs)
                val_outputs = [torch.sigmoid(x) > 0.5 for x in val_outputs]
                value = metric(y_pred=val_outputs, y=val_labels)

            metric_value = metric.aggregate().item()
            scheduler.step(metric_value)  # Scheduler update based on validation metric
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr}")

            if metric_value > best_metric:
                best_metric = metric_value
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), str(model_path))
                print(f"Saved new best metric model")
                counter = 0
            else:
                counter += 1

            print(f"Current mean Dice: {metric_value:.4f}, Best mean Dice: {best_metric:.4f}, at epoch {best_metric_epoch}")

            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print("Training complete.")
