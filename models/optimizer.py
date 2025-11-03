"""
optimizer.py

This module provides functionality to configure and retrieve an optimizer for training
3D deep learning models, supporting optimizers like L-BFGS and Adam based on the user-defined configuration.

Dependencies:
    - yacs.config: Manages hierarchical configuration settings.
    - torch: Supplies model parameter access and optimizer implementations.
    - models.networks: Contains the model definitions such as UNETDIP.
    - typing: Enables type hinting for function arguments and return types.

Owner:
    Edwing Ulin

Version:
    v1.0.1
"""

from typing import Union
import torch
import yacs.config
from models.networks import FCDenseNet

def get_optimizer(
        model: Union[FCDenseNet, torch.nn.DataParallel],
        cfg: yacs.config.CfgNode
) -> torch.optim.Optimizer:

    """
    Get an optimizer and its associated closure (if required, e.g., for L-BFGS).

    Parameters
    ----------
    model : Union[FastSurferCNN, FastSurferVINN, torch.nn.DataParallel]
        The model to optimize.
    cfg : yacs.config.CfgNode
        Configuration Node.

    Returns
    -------
    torch.optim.Optimizer
        The optimizer.

    Raises
    ------
    NotImplementedError
        If the optimizer is not supported.
    """
    if cfg.OPTIMIZER.OPTIMIZING_METHOD == "lbfgs":

        return torch.optim.LBFGS(
            model.parameters(),
            lr = cfg.OPTIMIZER.BASE_LR,
            max_iter = cfg.OPTIMIZER.MAX_ITER,
            history_size= cfg.OPTIMIZER.HISTORY_SIZE,
            line_search_fn=None,
        )
    elif cfg.OPTIMIZER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.OPTIMIZER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            f"Optimizer '{cfg.OPTIMIZER.OPTIMIZING_METHOD}' is not supported."
        )