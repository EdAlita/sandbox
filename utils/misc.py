"""
misc.py

This module contains various utility functions used throughout the project, including:
- Experiment management (finding the latest experiment)
- Medical imaging file handling (NIfTI reading)
- Tensor statistics computation
- Image visualization and logging with TensorBoard
- File processing utilities

Dependencies:
-------------
- `nibabel`: Handles medical imaging files (NIfTI format).
- `yacs`: Manages configuration settings.
- `torch`: Deep learning framework.
- `numpy`: Numerical computations.
- `re`: Regular expressions for file name pattern matching.
- `matplotlib.pyplot`: Visualization of medical images.
- `pathlib.Path`: Simplifies file path manipulations.

Author:
-------
Edwing Ulin

Version:
--------
v1.0.1
"""
from io import BytesIO
import numpy as np
from typing import Optional
from click import Option
import torch
from torch.utils.tensorboard import SummaryWriter
import yacs.config
import matplotlib.pyplot as plt
from skimage import color
from torchvision import utils
import torch
from torchvision.utils import make_grid

import data.loader

def plot_predictions(
    images_batch: torch.Tensor,
    labels_batch: torch.Tensor,
    batch_output: torch.Tensor,
    plt_title: str,
    file_save_name: str,
    writer: Optional[SummaryWriter] = None,
    step: Optional[int] = None,
    type: Optional[str] = None,
) -> None:
    """
    Plot predictions from validation set.

    Parameters
    ----------
    images_batch : torch.Tensor
        Batch of images.
    labels_batch : torch.Tensor
        Batch of labels.
    batch_output : torch.Tensor
        Batch of output.
    plt_title : str
        Plot title.
    file_save_name : str
        Name the plot should be saved tp.
    """
    f = plt.figure(figsize=(20, 10))
    n, c, h, w = images_batch.shape
    mid_slice = c // 2
    images_batch = torch.unsqueeze(images_batch[:, mid_slice, :, :], 1)
    img_grid = utils.make_grid(images_batch.cpu(), nrow=4)

    plt.subplot(311)
    gt_grid = utils.make_grid(labels_batch.unsqueeze_(1).cpu(), nrow=4)[0]
    gt_color = color.label2rgb(gt_grid.numpy(), bg_label=0)
    plt.imshow(img_grid.numpy().transpose((1, 2, 0)))
    plt.imshow(gt_color, alpha=0.5)
    plt.title("Ground Truth", fontsize=24)
    plt.axis("off")

    pred_grid = utils.make_grid(batch_output.unsqueeze_(1).cpu(), nrow=4)[0]
    pred_color = color.label2rgb(pred_grid.numpy(), bg_label=0)
    plt.subplot(312)
    plt.imshow(img_grid.numpy().transpose((1, 2, 0)))
    plt.imshow(pred_color, alpha=0.5)
    plt.title("Prediction", fontsize=24)
    plt.axis("off")

    plt.subplot(313)
    diff = (gt_grid != pred_grid)
    # Ensure numpy array on CPU
    if isinstance(diff, torch.Tensor):
        diff_np = diff.cpu().float().numpy()
    else:
        diff_np = np.array(diff, dtype=float)

    # Overlay the difference map on the input image for visibility.
    img_display = img_grid.numpy().transpose((1, 2, 0))
    if diff_np.max() == 0:
        plt.imshow(img_display)
        plt.text(
            img_display.shape[1] / 2,
            img_display.shape[0] / 2,
            "No differences detected",
            color="white",
            fontsize=20,
            ha="center",
            va="center",
        )
    else:
        plt.imshow(img_display)
        plt.imshow(diff_np, cmap="jet", alpha=0.6, vmin=0, vmax=1)

    plt.title("Difference Map", fontsize=24)
    plt.axis("off")

    plt.suptitle(plt_title, fontsize=32)
    plt.tight_layout()

    if writer is not None:

        buf = BytesIO()
        f.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)

        image = plt.imread(buf)

        writer.add_image(f"{type} Results", image.transpose((2, 0, 1)), global_step=step)
        buf.close()
    
    f.savefig(file_save_name, bbox_inches="tight")  
    
    plt.close(f)
    plt.gcf().clear()

def update_num_steps(
    dataloader: data.loader.DataLoader, cfg: yacs.config.CfgNode
):
    """
    Update the number of steps.

    Parameters
    ----------
    dataloader : FastSurferCNN.data_loader.loader.DataLoader
        [MISSING].
    cfg : yacs.config.CfgNode
        [MISSING].
    """
    cfg.TRAIN.NUM_STEPS = len(dataloader)

def plot_input_summary(dict_batch, writer, global_step):
    """
    Create a matplotlib figure with image, label, and weight grids,
    then add it to TensorBoard SummaryWriter.

    Parameters
    ----------
    dict_batch : dict
        Dictionary containing "image", "label", and "weight" tensors.
        Shapes:
            image  -> (N, C, H, W)
            label  -> (N, H, W) or (N, 1, H, W)
            weight -> (N, H, W) or (N, 1, H, W)
    writer : torch.utils.tensorboard.SummaryWriter
        TensorBoard writer instance.
    global_step : int
        Current training step for logging.
    """
    images_batch = dict_batch["image"]
    labels_batch = dict_batch["label"]
    weights_batch = dict_batch["weight"]
    n, c, h, w = images_batch.shape
    mid_slice = c // 2
    images_batch = torch.unsqueeze(images_batch[:, mid_slice, :, :], 1)
    img_grid = utils.make_grid(images_batch.cpu(), nrow=4)

    gt_grid = utils.make_grid(labels_batch.unsqueeze_(1).cpu(), nrow=4)[0]
    gt_color = color.label2rgb(gt_grid.numpy(), bg_label=0)

    weight_grid = utils.make_grid(weights_batch.unsqueeze_(1).cpu(), nrow=4)[0]
    wgt_color = color.label2rgb(weight_grid.numpy(), bg_label=0)

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(20, 10))
    axes[0].imshow(img_grid.numpy().transpose((1, 2, 0)))
    axes[0].set_title("Input Images", fontsize=24)
    axes[1].imshow(gt_color)
    axes[1].set_title("Labels", fontsize=24)
    axes[2].imshow(wgt_color)
    axes[2].set_title("Weights", fontsize=24)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)

    image = plt.imread(buf)

    # Add to TensorBoard
    writer.add_image("Inputs", image.transpose((2, 0, 1)), global_step)

    # Close figure to free memory
    buf.close()
    plt.close(fig)
