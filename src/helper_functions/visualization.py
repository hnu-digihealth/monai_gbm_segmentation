# Python Standard Library
# Local Libraries
import logging
import os

# Third Party Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from monai.data import DataLoader
from numpy import uint8

logger = logging.getLogger("Visualization")

logger.info("Visualization module loaded")


def save_visualizations(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    original_files: list,
) -> None:
    """Save visualizations of model predictions.

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
    logger.info(f"Output directory created: {output_dir}")
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inputs = batch["img"].to(device)
            labels = batch["seg"].to(device)
            logger.debug(f"Processing batch {i} with {inputs.shape[0]} samples")

            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5

            for j in range(inputs.shape[0]):
                # fig, axs = plt.subplots(1, 4, figsize=(24, 6))

                # axs[0].set_title("Original Image")
                original_img_path = original_files[i * inputs.shape[0] + j]["img"]
                original_img = cv2.imread(original_img_path)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB).astype(uint8)
                # axs[0].imshow(original_img)
                plt.imshow(original_img)
                plt.savefig(os.path.join(output_dir, "1.png"))

                # axs[0].imshow(original_img)

                # axs[1].set_title("Normalized Image")
                img = inputs[j].cpu().permute(1, 2, 0).astype(uint8)
                # axs[1].imshow(img)
                fig = plt.imshow(img)
                plt.savefig(os.path.join(output_dir, "2.png"))

                # axs[2].set_title("Ground Truth Mask")
                gt_mask = labels[j][0].cpu().astype(uint8)
                print(type(gt_mask))
                print(np.unique(gt_mask))
                gt_mask = np.invert(gt_mask)
                print(type(gt_mask))
                print(np.unique(gt_mask))
                # axs[2].imshow(gt_mask, cmap='gray', vmin=254, vmax=255)
                fig = plt.imshow(gt_mask, cmap="gray", vmin=254, vmax=255)
                plt.savefig(os.path.join(output_dir, "3.png"))

                # axs[3].set_title("Predicted Mask")
                pred_mask = preds[j][0].cpu().astype(uint8)
                # axs[3].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
                fig = plt.imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
                plt.savefig(os.path.join(output_dir, "4.png"))

                # fig.savefig(os.path.join(output_dir, f"sample_{i}_{j}.png"))
                plt.close(fig)
