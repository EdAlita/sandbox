# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Optional

import matplotlib.pyplot as plt

# IMPORTS
import numpy as np
import torch
import yacs.config

from utils.gen_logger import loggen
from utils.metrics import DiceScore
from utils.misc import plot_predictions

logger = loggen(__name__)


class Meter:
    """[MISSING]."""

    def __init__(
        self,
        cfg: yacs.config.CfgNode,
        mode: str,
        global_step: int,
        total_iter: Optional[int] = None,
        total_epoch: Optional[int] = None,
        class_names: Optional[Any] = None,
        device: Optional[Any] = None,
        writer: Optional[Any] = None,
    ):
        """
        Construct a Meter object.

        Parameters
        ----------
        cfg
            [MISSING]
        mode
            [MISSING]
        global_step
            [MISSING]
        total_iter
            [MISSING]
        total_epoch
            [MISSING]
        class_names
            [MISSING]
        device
            [MISSING]
        writer
            [MISSING]

        """
        self._cfg = cfg
        self.mode = mode.capitalize()
        self.confusion_mat = False
        self.class_names = class_names
        if self.class_names is None:
            self.class_names = [f"{c+1}" for c in range(cfg.MODEL.N_CLASSES)]

        self.batch_losses = []
        self.writer = writer
        self.global_iter = global_step
        self.total_iter_num = total_iter
        self.total_epochs = total_epoch
        self.dice_score = DiceScore(num_classes=cfg.MODEL.N_CLASSES, device=device)

    def reset(self):
        """
        Reset bach losses and dice scores.
        """
        self.batch_losses = []
        self.dice_score.reset()

    def enable_confusion_mat(self):
        """
        [MISSING].
        """
        self.confusion_mat = True

    def disable_confusion_mat(self):
        """
        [MISSING].
        """
        self.confusion_mat = False
    
    def update_stats(self, pred, labels, batch_loss):
        """
        [MISSING].
        """
        self.dice_score.update((pred, labels), self.confusion_mat)
        self.batch_losses.append(batch_loss.item())

    def write_summary(self, loss_total, lr=None, loss_ce=None, loss_dice=None):
        """
        Write a summary of the losses and scores.

        Parameters
        ----------
        loss_total : [MISSING]
            [MISSING].
        lr : default = None
            [MISSING] (Default value = None).
        loss_ce : default = None
            [MISSING] (Default value = None).
        loss_dice : default = None
            [MISSING] (Default value = None).
        """
        self.writer.add_scalar(
            f"{self.mode}/total_loss", loss_total.item(), self.global_iter
        )
        if self.mode == "Train":
            self.writer.add_scalar("Train/lr", lr[0], self.global_iter)
            if loss_ce:
                self.writer.add_scalar(
                    "Train/ce_loss", loss_ce.item(), self.global_iter
                )
            if loss_dice:
                self.writer.add_scalar(
                    "Train/dice_loss", loss_dice.item(), self.global_iter
                )

        self.global_iter += 1

    def log_iter(self, cur_iter: int, cur_epoch: int):
        """
        Log the current iteration.

        Parameters
        ----------
        cur_iter : int
            Current iteration.
        cur_epoch : int
            Current epoch.
        """
        if (cur_iter + 1) % self._cfg.TRAIN.LOG_INTERVAL == 0:
            logger.info(
                "{} Epoch [{}/{}] Iter [{}/{}] with loss {:.4f}".format(
                    self.mode,
                    cur_epoch + 1,
                    self.total_epochs,
                    cur_iter + 1,
                    self.total_iter_num,
                    np.array(self.batch_losses).mean(),
                )
            )

    def log_epoch(self, cur_epoch: int):
        """
        Log the current epoch.

        Parameters
        ----------
        cur_epoch : int
            Current epoch.
        """
        dice_score = self.dice_score.compute_dsc()
        self.writer.add_scalar(f"{self.mode}/mean_dice_score", dice_score, cur_epoch)

    def plot_predictions(
        self,
        images_batch: torch.Tensor,
        labels_batch: torch.Tensor,
        batch_output: torch.Tensor,
        plt_title: str,
        file_save_name: str,
        test: Optional[bool] = True,
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
            Batch of network outputs.
        plt_title : str
            Title of the plot.
        file_save_name : str
            File name to save the plot.
        """
        if test:
            plot_predictions(
            images_batch,
            labels_batch,
            batch_output,
            plt_title,
            file_save_name)
        else:
            plot_predictions(
                images_batch,
                labels_batch,
                batch_output,
                plt_title,
                file_save_name,
                self.writer,
                self.global_iter,
                self.mode,
            )
    
