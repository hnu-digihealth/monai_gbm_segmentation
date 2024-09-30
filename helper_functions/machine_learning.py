# Third Party Libraries
import torch.nn as nn
from monai.losses import DiceLoss, FocalLoss
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score, accuracy_score

# Local Libraries
from helper_functions.visualization import save_visualizations


class CombinedDiceFocalLoss(nn.Module):
    """
    Custom loss class, combining Dice and Focal loss.
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
    def __init__(self, model, loss_fn, metric, val_loader, original_files, visualize_validation=False):
        super(UNetLightning, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.metric = metric
        self.val_loader = val_loader
        self.original_files = original_files
        self.visualize_validation = visualize_validation
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

        return {"val_dice": val_dice}

    def on_validation_epoch_end(self):
        """
        Actions to perform at the end of the validation epoch.
        """
        if self.visualize_validation:
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