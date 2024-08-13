import os
from pathlib import Path
import cv2
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensor, Resized, MapTransform
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer, LightningModule
from monai.losses import DiceLoss, FocalLoss

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

# Define the CombinedDiceFocalLoss class
class CombinedDiceFocalLoss(torch.nn.Module):
    """
    Combine Dice and Focal loss.
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

# Define the ConvertToSingleChannel class
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
            img_single_channel[img[0] == 0] = 0  # Tumor (Black)
            img_single_channel[img[0] == 255] = 1  # Healthy (White)
            d[key] = torch.tensor(img_single_channel, dtype=torch.float32).unsqueeze(0)
        return d

# Define the UNetLightning class with additional metrics
class UNetLightning(LightningModule):
    """
    LightningModule for UNet with additional metrics.
    """
    def __init__(self, model, loss_fn, metric, val_loader, original_files):
        super(UNetLightning, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.metric = metric
        self.val_loader = val_loader
        self.original_files = original_files

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["img"], batch["seg"]
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch["img"], batch["seg"]
        outputs = self(inputs)
        val_outputs = torch.sigmoid(outputs) > 0.5

        dice_value = self.metric(val_outputs, labels)
        dice_value = dice_value[~torch.isnan(dice_value)].mean()
        self.log("val_dice", dice_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        preds = val_outputs.cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()
        
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        iou = jaccard_score(labels, preds)
        accuracy = accuracy_score(labels, preds)

        self.log("val_precision", precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_recall", recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_iou", iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"val_dice": dice_value}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_dice',
                'frequency': 1,
            }
        }

    def test_step(self, batch, batch_idx):
        inputs, labels = batch["img"], batch["seg"]
        outputs = self(inputs)
        test_outputs = torch.sigmoid(outputs) > 0.5

        dice_value = self.metric(test_outputs, labels)
        dice_value = dice_value[~torch.isnan(dice_value)].mean()
        self.log("test_dice", dice_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        preds = test_outputs.cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()
        
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        iou = jaccard_score(labels, preds)
        accuracy = accuracy_score(labels, preds)

        self.log("test_precision", precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_recall", recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_iou", iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"test_dice": dice_value}

    def save_visualizations(self, epoch):
        self.model.eval()
        output_dir = f"visualizations_epoch_{epoch}"
        os.makedirs(output_dir, exist_ok=True)
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                inputs = batch['img'].to(self.device)
                labels = batch['seg'].to(self.device)
                
                outputs = self.model(inputs)
                preds = torch.sigmoid(outputs) > 0.5

                for j in range(inputs.shape[0]):
                    file_index = i * self.val_loader.batch_size + j
                    if file_index >= len(self.original_files):
                        continue
                    
                    original_img_path = self.original_files[file_index]['img']
                    original_img = cv2.imread(original_img_path)
                    if original_img is None:
                        print(f"Could not read image {original_img_path}")
                        continue
                        
                    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
                    
                    axs[0].set_title("Original Image")
                    axs[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
                    
                    axs[1].set_title("Transformed Image")
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

if __name__ == "__main__":
    # Define paths for the test data
    base_dir = Path(__file__).resolve().parent.parent
    test_images_path = base_dir.parent / 'Data' / 'test_data_t3' / 'test3' / 'img'
    test_labels_path = base_dir.parent / 'Data' / 'test_data_t3' / 'test3' / 'lbl'

    test_images = sorted([x for x in test_images_path.iterdir() if x.suffix == '.png'])
    test_labels = sorted([x for x in test_labels_path.iterdir() if x.suffix == '.png'])

    # Create the test data
    test_files = [{'img': img, 'seg': seg} for img, seg in zip(test_images, test_labels)]

    # Define the transformations
    test_transforms = Compose([
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        HENormalization(keys=["img"], target_mean=(0.485, 0.456, 0.406), target_std=(0.229, 0.224, 0.225)),
        ConvertToSingleChannel(keys=["seg"]),
        Resized(keys=["img", "seg"], spatial_size=(1024, 1024)),
        ScaleIntensityd(keys=["img"]),
        ToTensor()
    ])

    # Create the test dataset and dataloader
    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=16, num_workers=8, persistent_workers=True, pin_memory=torch.cuda.is_available())

    # Load the trained model from the checkpoint
    checkpoint_path = base_dir / 'Modelle' / 'model2.ckpt'
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

    loss_function = CombinedDiceFocalLoss(dice_weight=0.7, focal_weight=0.3)
    metric = DiceMetric(include_background=True, reduction="mean")

    pl_model = UNetLightning(model, loss_function, metric, test_loader, test_files)

    # Initialize the trainer
    trainer = Trainer(devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu")

    # Evaluate the model on the test data
    trainer.test(pl_model, test_loader, ckpt_path=checkpoint_path)

    # Save visualizations
    pl_model.save_visualizations(epoch=25)
