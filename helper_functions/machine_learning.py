# Third Party Libraries
import torch.nn as nn
import numpy as np
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, MeanIoU, GeneralizedDiceScore
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score


# Local Libraries
from helper_functions.visualization import save_visualizations


class UNetLightning(pl.LightningModule):
    """
    PyTorch Lightning module for UNet model.

    Args:
        model (torch.nn.Module): The UNet model.
        loss_fn (callable): Loss function.
        metric (callable): Metric for evaluation.
        data_loader (torch.utils.data.DataLoader): Data loader used to compute test/val metrics.
            Should be the test or validation data loader, depending on run.
        original_files (list): List of original file paths for visualization.
    """
    def __init__(
            self, 
            model,   
            val_loader, 
            original_files,
            loss_fn=DiceFocalLoss(sigmoid=True, lambda_dice=0.7, lambda_focal=0.3), 
            #metric=GeneralizedDiceScore(include_background=False, reduction="mean_batch"),
            metric=DiceMetric(include_background=False, reduction="mean_batch", num_classes=2),
            metric_iou=MeanIoU(include_background=False, reduction="mean_batch"),
            visualize_validation=False
        ):
        super(UNetLightning, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.metric = metric
        self.metric_iou = metric_iou
        self.val_loader = val_loader
        self.original_files = original_files
        self.visualize_validation = visualize_validation
        self.f1_step = []
        self.iou_step = []
        self.recall_step = []
        self.precision_step = []

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
        
        # Calculate and log training Dice metric
        # train_outputs = torch.sigmoid(outputs) > 0.5
        # self.metric(train_outputs, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
       
        return loss

    # def on_train_epoch_end(self):
    #     print(self.metric)
    #     print(self.metric.aggregate())
    #     dice = self.metric.aggregate().item()
    #     self.metric.reset()
    #     self.log("train_dice", dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)

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
        loss = self.loss_fn(outputs, labels)
        val_outputs = torch.sigmoid(outputs) > 0.5

        self.metric(val_outputs, labels)
        self.metric_iou(val_outputs, labels)
        
        self.calculate_metrics(val_outputs, labels)
    
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        """
        Actions to perform at the end of the validation epoch.
        """
        if self.visualize_validation:
            save_visualizations(self.model, self.val_loader, self.device, self.current_epoch, self.original_files)
        dice = self.metric.aggregate().item()
        monai_iou = self.metric_iou.aggregate().item()

        self.log("val_dice", dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", np.nanmean(self.f1_step), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_iou_monai", monai_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_iou", np.nanmean(self.iou_step), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_recall", np.nanmean(self.recall_step), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_precision", np.nanmean(self.precision_step), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.f1_step.clear()
        self.iou_step.clear()
        self.recall_step.clear()
        self.precision_step.clear()
        self.metric.reset()
        self.metric_iou.reset()

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

        self.metric(test_outputs, labels)
        
        self.calculate_metrics(test_outputs, labels)
    
    def on_test_epoch_end(self):
        dice = self.metric.aggregate().item()

        self.log("test_dice", dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1", np.nanmean(self.f1_step), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_iou", np.nanmean(self.iou_step), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_recall", np.nanmean(self.recall_step), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_precision", np.nanmean(self.precision_step), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.f1_step.clear()
        self.iou_step.clear()
        self.recall_step.clear()
        self.precision_step.clear()
        self.metric.reset()

    def calculate_metrics(self, test_outputs: torch.Tensor, labels: torch.Tensor) -> tuple:
        metrics = {'iou': [], 'f1': [], 'recall': [], 'precision': [], 'me_iou': [], 'me_f1': []}
        for i in range(len(test_outputs)):
            pred = test_outputs[i].cpu().numpy().flatten()
            label = labels[i].cpu().numpy().flatten()
            metrics['precision'].append(precision_score(label, pred, average='binary', zero_division=1, pos_label=1))
            metrics['recall'].append(recall_score(label, pred, average='binary', zero_division=1, pos_label=1))
            metrics['iou'].append(jaccard_score(label, pred, average='binary', zero_division=1, pos_label=1))
            metrics['f1'].append(f1_score(label, pred, average='binary', zero_division=1, pos_label=1))

        self.iou_step.append(np.nanmean(metrics['iou']))
        self.f1_step.append(np.nanmean(metrics['f1']))
        self.recall_step.append(np.nanmean(metrics['recall']))
        self.precision_step.append(np.nanmean(metrics['precision']))
