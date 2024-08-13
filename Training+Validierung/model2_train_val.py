import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensor, 
    RandRotate90d, RandFlipd, MapTransform, RandAdjustContrastd, 
    RandGaussianNoised, RandZoomd, RandAffined
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, FocalLoss
from monai.metrics import DiceMetric
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import matplotlib
matplotlib.use('Agg')

# Set the float32 matrix multiplication precision
torch.set_float32_matmul_precision('medium')

# Define the device variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def save_visualizations(model, data_loader, device, epoch, original_files):
    """
    Save visualizations of model predictions.

    Args:
        model (nn.Module): The model.
        data_loader (DataLoader): DataLoader for the data.
        device (torch.device): Device to use for computation.
        epoch (int): Current epoch number.
        original_files (list): List of original files.
    """
    model.eval()
    output_dir = f"visualizations_epoch_{epoch}"
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inputs = batch['img'].to(device)
            labels = batch['seg'].to(device)
            
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5

            for j in range(inputs.shape[0]):
                fig, axs = plt.subplots(1, 4, figsize=(24, 6))
                
                axs[0].set_title("Original Image")
                original_img_path = original_files[i * inputs.shape[0] + j]['img']
                original_img = cv2.imread(original_img_path)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                axs[0].imshow(original_img)
                
                axs[1].set_title("Normalized Image")
                img = inputs[j].cpu().permute(1, 2, 0).numpy()
                axs[1].imshow(img)
                
                axs[2].set_title("Ground Truth Mask")
                gt_mask = labels[j][0].cpu().numpy()
                axs[2].imshow(gt_mask, cmap='gray')
                
                axs[3].set_title("Predicted Mask")
                pred_mask = preds[j][0].cpu().numpy()
                axs[3].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
                
                fig.savefig(os.path.join(output_dir, f"sample_{i}_{j}.png"))
                plt.close(fig)

# Define paths for training and validation data
base_dir = Path(__file__).resolve().parent.parent

# Define paths for training and validation data
data_path = base_dir.parent / 'Data' / 'training_data_1k'
data_path_val = base_dir.parent / 'Data' / 'test_data_t3'

train_images_path = data_path / 'train' / 'img'
train_labels_path = data_path / 'train' / 'lbl'
val_images_path = data_path_val / 'validate3' / 'img'
val_labels_path = data_path_val / 'validate3' / 'lbl'

train_images = sorted([x for x in train_images_path.iterdir() if x.suffix == '.png'])
train_labels = sorted([x for x in train_labels_path.iterdir() if x.suffix == '.png'])
val_images = sorted([x for x in val_images_path.iterdir() if x.suffix == '.png'])
val_labels = sorted([x for x in val_labels_path.iterdir() if x.suffix == '.png'])

# Convert Paths to strings
train_images = [str(x) for x in train_images]
train_labels = [str(x) for x in train_labels]
val_images = [str(x) for x in val_images]
val_labels = [str(x) for x in val_labels]

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

train_loader = DataLoader(train_ds, batch_size=16, num_workers=8, persistent_workers=True, shuffle=True, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_ds, batch_size=16, num_workers=8, persistent_workers=True, pin_memory=torch.cuda.is_available())

class UNetLightning(pl.LightningModule):
    """
    PyTorch Lightning module for UNet model.

    Args:
        model (torch.nn.Module): The UNet model.
        loss_fn (callable): Loss function.
        metric (callable): Metric for evaluation.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        original_files (list): List of original file paths for visualization.
    """
    def __init__(self, model, loss_fn, metric, val_loader, original_files):
        super(UNetLightning, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.metric = metric
        self.val_loader = val_loader
        self.original_files = original_files
        self.validation_step_outputs = []

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.

        Args:
            batch (dict): Dictionary containing input and target tensors.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value for the batch.
        """
        inputs, labels = batch["img"], batch["seg"]
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Calculate and log training Dice metric
        train_outputs = torch.sigmoid(outputs) > 0.5
        train_dice = self.metric(train_outputs, labels)
        train_dice = train_dice[~torch.isnan(train_dice)].mean()  # Remove NaNs and calculate mean
        self.log("train_dice", train_dice, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.

        Args:
            batch (dict): Dictionary containing input and target tensors.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Dice metric value for the batch.
        """
        inputs, labels = batch["img"], batch["seg"]
        outputs = self(inputs)
        val_outputs = torch.sigmoid(outputs) > 0.5
        val_dice = self.metric(val_outputs, labels)
        val_dice = val_dice[~torch.isnan(val_dice)].mean()  # Remove NaNs and calculate mean
        self.log("val_dice", val_dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return val_dice

    def on_validation_epoch_end(self):
        """
        Actions to perform at the end of the validation epoch.
        """
        save_visualizations(self.model, self.val_loader, self.device, self.current_epoch, self.original_files)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """
        Configure the optimizers and learning rate scheduler.

        Returns:
            dict: Dictionary containing the optimizer and scheduler configurations.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_dice',
                'frequency': 1,
            }
        }

if __name__ == "__main__":
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

    loss_function = CombinedDiceFocalLoss(dice_weight=0.7, focal_weight=0.3)
    metric = DiceMetric(include_background=True, reduction="mean")

    pl_model = UNetLightning(model, loss_function, metric, val_loader, val_files)
    
    trainer = pl.Trainer(
        max_epochs=100,
        devices=1, accelerator="gpu" if torch.cuda.is_available() else 0,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val_dice', mode='max', save_top_k=1, verbose=True),
            pl.callbacks.EarlyStopping(monitor='val_dice', patience=10, mode='max', verbose=True)
        ]
    )
    trainer.fit(pl_model, train_loader, val_loader)
