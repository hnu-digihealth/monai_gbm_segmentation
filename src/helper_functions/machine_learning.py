"""
Model module for MONAI-based UNet using PyTorch Lightning.

Defines the UNetLightning wrapper with integrated loss, metrics, training,
validation, and optional visualization functionality.
"""

# Python Standard Libraries
import logging
from pathlib import Path
from typing import Any

# Third Party Libraries
import pytorch_lightning as pl
import torch
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Local Libraries
from src.helper_functions.visualization import save_visualizations

logger = logging.getLogger("MachineLearning")


class UNetLightning(pl.LightningModule):
    """
    PyTorch Lightning module encapsulating the UNet model, loss, metrics, and evaluation logic.

    Args:
        val_loader (DataLoader): Validation or test data loader (used for metrics and visualization).
        original_files (list[Path]): Original file paths used for saving visualizations.
        loss_fn (callable, optional): Loss function to use. Defaults to DiceFocalLoss.
        metric (callable, optional): Metric for F1/Dice evaluation. Defaults to MONAI DiceMetric.
        metric_iou (callable, optional): Metric for IoU evaluation. Defaults to MONAI MeanIoU.
        visualize_validation (bool): If True, saves validation visualizations using `save_visualizations()`.
    """

    def __init__(
        self,
        val_loader: DataLoader,
        original_files: list[dict[str, Path]],
        loss_fn: torch.nn.Module | None = None,
        metric: DiceMetric | None = None,
        metric_iou: MeanIoU | None = None,
        visualize_validation: bool = False,
    ) -> None:
        super(UNetLightning, self).__init__()
        logger.info("Initializing UNetLightning model")

        # Initialize model and components
        self.model = init_unet_model()

        self.loss_fn = (
            loss_fn if loss_fn is not None else DiceFocalLoss(sigmoid=True, lambda_dice=0.7, lambda_focal=0.3)
        )

        self.metric_f1 = (
            metric
            if metric is not None
            else DiceMetric(
                include_background=False,
                reduction="mean_batch",
                num_classes=2,
                ignore_empty=False,
            )
        )

        self.metric_iou = (
            metric_iou
            if metric_iou is not None
            else MeanIoU(
                include_background=False,
                reduction="mean_batch",
                ignore_empty=False,
            )
        )

        self.val_loader = val_loader
        self.original_files = original_files
        self.visualize_validation = visualize_validation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor from the model.

        """
        logger.debug("Running forward pass")
        return self.model(x)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for a single batch.

        Args:
            batch (dict): Dictionary containing input and target tensors.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value for the batch.

        """
        inputs, labels = batch["img"], batch["seg"]
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        logger.debug(f"Training batch {batch_idx}, loss: {loss.item():.4f}")
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, float]:
        """Validation step for a single batch.

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

    def on_validation_epoch_end(self) -> None:
        """Actions to perform at the end of the validation epoch."""
        if self.visualize_validation:
            logger.info(f"Saving validation visualizations for epoch {self.current_epoch}")
            save_visualizations(
                self.model,
                self.val_loader,
                self.device,
                self.current_epoch,
                self.original_files,
            )
        dice = self.metric_f1.aggregate().item()
        iou = self.metric_iou.aggregate().item()
        logger.info(f"Validation results after epoch {self.current_epoch}: Dice = {dice:.4f}, IoU = {iou:.4f}")

        self.log("val_dice", dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_iou", iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.metric_f1.reset()
        self.metric_iou.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure the optimizers and learning rate scheduler.

        Returns:
            dict: Dictionary containing the optimizer and scheduler configurations.

        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_dice",
                "frequency": 1,
            },
        }

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, float]:
        inputs, labels = batch["img"], batch["seg"]
        outputs = self(inputs)
        test_outputs = torch.sigmoid(outputs) > 0.5

        self.metric_f1(test_outputs, labels)
        self.metric_iou(test_outputs, labels)

    def on_test_epoch_end(self) -> None:
        if self.visualize_validation:
            logger.info(f"Saving test visualizations for epoch {self.current_epoch}")
            save_visualizations(
                self.model,
                self.val_loader,
                self.device,
                self.current_epoch,
                self.original_files,
            )

        dice = self.metric_f1.aggregate().item()
        iou = self.metric_iou.aggregate().item()
        logger.info(f"Test results after epoch {self.current_epoch}: Dice = {dice:.4f}, IoU = {iou:.4f}")

        self.log("test_dice", dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_iou", iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.metric_f1.reset()
        self.metric_iou.reset()


def init_unet_model() -> UNet:
    return UNet(
        spatial_dims=2,
        in_channels=3,  # 3 for RGB images
        out_channels=1,  # 1 for binary segmentation
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,  # Number of residual units in each layer
        norm=Norm.BATCH,  # Batch normalization
        dropout=0.1,
    )
