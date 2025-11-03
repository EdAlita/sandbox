"""
lr_scheduler.py

This script provides the learning rate scheduler to use during model training.
It supports several scheduler types based on configuration, including step LR,
cosine annealing with warm restarts, and multi-step LR.

Dependencies:
- torch.optim: Optimizer and scheduler utilities.
- yacs.config: Configuration management for training settings.

Owner:
Edwing Ulin

Version:
v1.0.0
"""

from typing import Union
import torch.optim
import torch.optim.lr_scheduler as scheduler
import yacs.config
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, MultiStepLR, ReduceLROnPlateau


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: yacs.config.CfgNode
) -> Union[None, StepLR, CosineAnnealingWarmRestarts, MultiStepLR, ReduceLROnPlateau]:
    """
    Returns the learning rate scheduler based on configuration settings.

    Args:
    -----
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        cfg (yacs.config.CfgNode): Configuration settings.

    Returns:
    --------
        Union[None, StepLR, CosineAnnealingWarmRestarts, MultiStepLR, ReduceLROnPlateau]:
            The configured learning rate scheduler or None if no scheduler is required.

    Raises:
    -------
        ValueError: If the specified scheduler type in the config is unsupported.
    """
    scheduler_type = cfg.OPTIMIZER.LR_SCHEDULER

    if scheduler_type == "step_lr":
        return scheduler.StepLR(
            optimizer=optimizer,
            step_size=cfg.OPTIMIZER.STEP_SIZE,
            gamma=cfg.OPTIMIZER.GAMMA
        )
    elif scheduler_type == "cosineWarmRestarts":
        return scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=cfg.OPTIMIZER.T_ZERO,
            T_mult=cfg.OPTIMIZER.T_MULT,
            eta_min=cfg.OPTIMIZER.ETA_MIN
        )
    elif scheduler_type == "multiStep":
        return scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=cfg.OPTIMIZER.MILESTONE,
            gamma=cfg.OPTIMIZER.GAMMA
        )
    elif scheduler_type == "reduceLROnPlateau":
        return scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',  # For loss minimization
            factor=cfg.OPTIMIZER.GAMMA,  # Factor by which to reduce LR
            patience=cfg.OPTIMIZER.PATIENCE,  # Number of epochs to wait
            min_lr=cfg.OPTIMIZER.ETA_MIN,  # Minimum learning rate
            verbose=True  # Print when LR is reduced
        )
    elif scheduler_type in ["NoScheduler", None]:
        return None
    else:
        raise ValueError(f"Scheduler type '{scheduler_type}' is not supported.")