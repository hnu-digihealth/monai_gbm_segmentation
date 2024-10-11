# Python Standard Library
import os

# Third Party Libraries
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from monai.data import DataLoader
from numpy import uint8


def save_visualizations(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    original_files: list
) -> None:
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
                original_img_path = original_files[i *
                                                   inputs.shape[0] + j]['img']
                original_img = cv2.imread(original_img_path)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB).astype(uint8)
                axs[0].imshow(original_img)

                #axs[0].imshow(original_img)

                #axs[1].set_title("Normalized Image")
                img = inputs[j].cpu().permute(1, 2, 0).astype(uint8)
                axs[1].imshow(img)

                axs[2].set_title("Ground Truth Mask")
                gt_mask = labels[j][0].cpu().astype(uint8)
                axs[2].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)

                axs[3].set_title("Predicted Mask")
                pred_mask = preds[j][0].cpu().astype(uint8)
                axs[3].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)

                fig.savefig(os.path.join(output_dir, f"sample_{i}_{j}.png"))
                plt.close(fig)
