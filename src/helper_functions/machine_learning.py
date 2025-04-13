# Third Party Libraries
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Local Libraries
from src.helper_functions.visualization import save_visualizations


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
            val_loader,
            original_files,
            loss_fn=DiceFocalLoss(sigmoid=True, lambda_dice=0.7, lambda_focal=0.3),
            metric=DiceMetric(include_background=False, reduction="mean_batch", num_classes=2, ignore_empty=False),
            metric_iou=MeanIoU(include_background=False, reduction="mean_batch", ignore_empty=False),
            visualize_validation=False
        ):
        super(UNetLightning, self).__init__()
        self.model = init_unet_model()
        self.loss_fn = loss_fn
        self.metric_f1 = metric
        self.metric_iou = metric_iou
        self.val_loader = val_loader
        self.original_files = original_files
        self.visualize_validation = visualize_validation

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

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

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
        loss = self.loss_fn(outputs, labels)
        val_outputs = torch.sigmoid(outputs) > 0.5

        self.metric_f1(val_outputs, labels)
        self.metric_iou(val_outputs, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        """
        Actions to perform at the end of the validation epoch.
        """
        if self.visualize_validation:
            save_visualizations(self.model, self.val_loader, self.device, self.current_epoch, self.original_files)
        dice = self.metric_f1.aggregate().item()
        iou = self.metric_iou.aggregate().item()

        self.log("val_dice", dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_iou", iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.metric_f1.reset()
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

        self.metric_f1(test_outputs, labels)
        self.metric_iou(test_outputs, labels)

    def on_test_epoch_end(self):
        if self.visualize_validation:
            save_visualizations(self.model, self.val_loader, self.device, self.current_epoch, self.original_files)

        dice = self.metric_f1.aggregate().item()
        iou = self.metric_iou.aggregate().item()

        self.log("test_dice", dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_iou", iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.metric_f1.reset()
        self.metric_iou.reset()


def init_unet_model():
    return UNet(
        spatial_dims=2,
        in_channels=3,  # 3 for RGB images
        out_channels=1,  # 1 for binary segmentation
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,  # Number of residual units in each layer
        norm=Norm.BATCH,  # Batch normalization
        dropout=0.1
    )
