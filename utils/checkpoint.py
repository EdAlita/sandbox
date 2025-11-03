import os
import torch
import yacs.config

from utils.gen_logger import loggen
from typing import Optional, Union

Scheduler = "torch.optim.lr_scheduler"

logger = loggen(__name__)


def create_checkpoint_dir(expr_dir: Union[os.PathLike], expr_num: int):
    """
    Create the checkpoint dir if not exists.

    Parameters
    ----------
    expr_dir : Union[os.PathLike]
        Directory to create.
    expr_num : int
        Number of expr [MISSING].

    Returns
    -------
    checkpoint_dir
        Directory of the checkpoint.
    """
    checkpoint_dir = os.path.join(expr_dir, "checkpoints", str(expr_num))
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    best_metric,
    num_gpus: int,
    cfg: yacs.config.CfgNode,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Scheduler] = None,
    best: bool = False,
) -> None:
    """
    Save the state of training for resume or fine-tune.

    Parameters
    ----------
    checkpoint_dir : str
        Path to the checkpoint directory.
    epoch : int
        Current epoch.
    best_metric : best_metric
        Best calculated metric.
    num_gpus : int
        Number of used gpus.
    cfg : yacs.config.CfgNode
        Configuration node.
    model : torch.nn.Module
        Used network model.
    optimizer : torch.optim.Optimizer
        Used network optimizer.
    scheduler : Optional[Scheduler]
        Used network scheduler. Optional (Default value = None).
    best : bool
        Whether this was the best checkpoint so far [MISSING] (Default value = False).
    """
    save_name = f"Epoch_{epoch:05d}_training_state.pkl"
    saving_model = model.module if num_gpus > 1 else model
    checkpoint = {
        "model_state": saving_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "config": cfg.dump(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_dir + "/" + save_name)

    if best:
        remove_ckpt(checkpoint_dir + "/Best_training_state.pkl")
        torch.save(checkpoint, checkpoint_dir + "/Best_training_state.pkl")


def remove_ckpt(ckpt: str):
    """
    Remove the checkpoint.

    Parameters
    ----------
    ckpt : str
        Path and filename to the checkpoint.
    """
    try:
        os.remove(ckpt)
    except FileNotFoundError:
        pass