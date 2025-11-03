import logging
import os
from config.defaults import get_cfg_defaults
# IMPORTS
from pprint import pprint
import time
from collections import defaultdict
from typing import Union

import numpy as np
import torch
from torchvision.utils import make_grid
import torch.optim.lr_scheduler as scheduler
import yacs.config
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import loader
from models.losses import get_loss_func
from models.networks import build_model
from models.optimizer import get_optimizer

from utils.gen_logger import loggen
from utils.lr_scheduler import get_lr_scheduler
from utils.meters import Meter
from utils.metrics import iou_score, precision_recall
import utils.checkpoint as cp
from utils.misc import plot_input_summary, update_num_steps
from torchsummary import summary

logger = loggen(__name__)

class Trainer:

    def __init__(self, cfg: yacs.config.CfgNode):
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        self.cfg = cfg
        logger.info(f"Training with config:")
        logger.info(cfg.dump())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(cfg)
        self.loss_func = get_loss_func(cfg)
        self.checkpoint_dir = cp.create_checkpoint_dir(cfg.LOG_DIR, cfg.EXPR_NUM)
        self.a = "{}\t" * (cfg.MODEL.N_CLASSES) + "{}"
        self.num_classes = cfg.MODEL.N_CLASSES
        self.plot_dir = os.path.join(cfg.LOG_DIR, "pred", str(cfg.EXPR_NUM))
        os.makedirs(self.plot_dir, exist_ok=True)
        self.class_names = ['WMH']

    def train(
        self,
        train_loader: loader.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Union[None, scheduler.StepLR, scheduler.CosineAnnealingWarmRestarts],
        train_meter: Meter,
        epoch,
    ) -> None:
        """
        Train the network to the given training data.

        Parameters
        ----------
        train_loader : loader.DataLoader
            Data loader for the training
        optimizer : torch.optim.optimizer.Optimizer
            Optimizer for the training
        scheduler : Union[None, scheduler.StepLR, scheduler.CosineAnnealingWarmRestarts]
            LR scheduler for the training.
        train_meter : Meter
            [MISSING].
        epoch : int
            [MISSING].

        """
        self.model.train()
        logger.info("Training started...")
        epoch_start = time.time()
        loss_batch = np.zeros(1)

        for curr_iter, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels, weights = (
                batch["image"].to(self.device),
                batch["label"].to(self.device),
                batch["weight"].to(self.device),
            )

  
            inputs = images

            optimizer.zero_grad()

            pred = self.model(inputs)
            loss_total, loss_dice, loss_ce = self.loss_func(pred, labels, weights)
            train_meter.update_stats(pred, labels, loss_total)
            train_meter.log_iter(curr_iter, epoch)
            if scheduler is not None:
                train_meter.write_summary(
                    loss_total, scheduler.get_last_lr(), loss_ce, loss_dice
                )
            else:
                train_meter.write_summary(
                    loss_total, [self.cfg.OPTIMIZER.BASE_LR], loss_ce, loss_dice
                )

            loss_total.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                #scheduler.step(epoch + curr_iter / len(train_loader))
            loss_batch += loss_total.item()

            if curr_iter % self.cfg.TRAIN.LOG_INTERVAL == 0 and curr_iter != 0:
                plt_title = "Epoch_" + str(epoch + 1) + "_Iter_" + str(curr_iter)

                file_save_name = os.path.join(
                    self.plot_dir,
                    "Epoch_" + str(epoch + 1) + "_Iter_" + str(curr_iter) + "_Training_Predictions.pdf",
                )

                _, batch_output = torch.max(pred, dim=1)
                train_meter.plot_predictions(
                    images, labels, batch_output, plt_title, file_save_name, test=False
                )

        train_meter.log_epoch(epoch)
        logger.info(
            "Training epoch {} finished in {:.04f} seconds".format(
                epoch+1, time.time() - epoch_start
            )
        )

    @torch.no_grad()
    def eval(
        self, val_loader: loader.DataLoader, val_meter: Meter, epoch: int
    ) -> np.ndarray:
        """
        Evaluate model and calculates stats.

        Parameters
        ----------
        val_loader : loader.DataLoader
            Value loader.
        val_meter : Meter
            Meter for the values.
        epoch : int
            Epoch to evaluate.

        Returns
        -------
        int, float, ndarray
            median miou [value].
        """
        logger.info(f"Evaluating model at epoch {epoch+1}...")
        self.model.eval()
        val_loss_total = 0.0
        val_loss_dice = 0.0
        val_loss_ce= 0.0
        ints_ = np.zeros(self.num_classes - 1)
        unis_ = np.zeros(self.num_classes - 1)
        miou = np.zeros(self.num_classes - 1)
        per_cls_counts_gt = np.zeros(self.num_classes - 1)
        per_cls_counts_pred = np.zeros(self.num_classes - 1)
        accs = np.zeros(self.num_classes - 1)
        # -1 to exclude background (still included in val loss)

        val_start = time.time()
        for curr_iter, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            images,labels, weights = (
                batch["image"].to(self.device),
                batch["label"].to(self.device),
                batch["weight"].to(self.device),
            )

            inputs = images
            pred = self.model(inputs)
            loss_total, loss_dice, loss_ce = self.loss_func(pred, labels, weights)

            val_loss_total += loss_total.item()
            val_loss_dice += loss_dice.item()
            val_loss_ce += loss_ce.item()

            _, batch_output = torch.max(pred, dim=1)

            int_, uni_ = iou_score(batch_output, labels, nclass=self.num_classes)
            ints_ += int_
            unis_ += uni_

            tpos, pcc_gt, pcc_pred = precision_recall(batch_output, labels, nclass=self.num_classes)
            accs += tpos
            per_cls_counts_gt += pcc_gt
            per_cls_counts_pred += pcc_pred


            if curr_iter % self.cfg.TRAIN.LOG_INTERVAL == 0 and curr_iter != 0:
                plt_title = "Validation Results Epoch " + str(epoch)

                file_save_name = os.path.join(
                    self.plot_dir,
                    "Epoch_" + str(epoch + 1) + "_Iter_" + str(curr_iter) + "_Validation_Predictions.pdf",
                )

                val_meter.plot_predictions(
                    images, labels, batch_output, plt_title, file_save_name, test=False
                )

            val_meter.update_stats(pred, labels, loss_total)
            val_meter.write_summary(loss_total)
            val_meter.log_iter(curr_iter, epoch)

        val_meter.log_epoch(epoch)
        logger.info(
            "Validation epoch {} finished in {:.04f} seconds".format(
                epoch, time.time() - val_start
            )
        )

        # Get final measures and log them
        
        ious = ints_ / unis_
        miou += ious
        val_loss_total /= curr_iter + 1
        val_loss_dice /= curr_iter + 1
        val_loss_ce /= curr_iter + 1

        # Log metrics
        logger.info(
            "[Epoch {} stats]: MIoU: {:.4f}; "
            "Mean Recall: {:.4f}; "
            "Mean Precision: {:.4f}; "
            "Avg loss total: {:.4f}; "
            "Avg loss dice: {:.4f}; "
            "Avg loss ce: {:.4f}".format(
                epoch,
                np.mean(ious),
                np.mean(accs / per_cls_counts_gt),
                np.mean(accs / per_cls_counts_pred),
                val_loss_total,
                val_loss_dice,
                val_loss_ce,
                )
            )

        return np.mean(np.mean(miou))

    def run(self):
        """
        Transfer the model to devices, create a tensor board summary writer and then perform the training loop.
        """
        if self.cfg.NUM_GPUS > 1:
            assert (
                self.cfg.NUM_GPUS <= torch.cuda.device_count()
            ), "Cannot use more GPU devices than available"
            print("Using ", self.cfg.NUM_GPUS, "GPUs!")
            self.model = torch.nn.DataParallel(self.model)

        val_loader = loader.get_dataloader(self.cfg, "val")
        train_loader = loader.get_dataloader(self.cfg, "train")

        dict = next(iter(train_loader))

        update_num_steps(train_loader, self.cfg)
        self.model = self.model.to(self.device)

        optimizer = get_optimizer(self.model, self.cfg)
        scheduler = get_lr_scheduler(optimizer, self.cfg)

        summary(self.model, input_size=(self.cfg.MODEL.NUM_CHANNELS, 240, 240))

        logger.info("Training from scratch")
        start_epoch = 0
        best_miou = 0

        logger.info(
            "{} parameters in total".format(
                sum(x.numel() for x in self.model.parameters())
            )
        )

        writer = SummaryWriter(self.cfg.SUMMARY_PATH, flush_secs=15)

        plot_input_summary(dict, writer, global_step=0)

        train_meter = Meter(
            self.cfg,
            mode="train",
            global_step=start_epoch * len(train_loader),
            total_iter=len(train_loader),
            total_epoch=self.cfg.TRAIN.NUM_EPOCHS,
            device=self.device,
            writer=writer,
        )

        val_meter = Meter(
            self.cfg,
            mode="val",
            global_step=start_epoch,
            total_iter=len(val_loader),
            total_epoch=self.cfg.TRAIN.NUM_EPOCHS,
            device=self.device,
            writer=writer,
        )

        logger.info("Summary path {}".format(self.cfg.SUMMARY_PATH))
        logger.info("Start epoch: {}".format(start_epoch + 1))

        best_epoch = -1
        early_stopping_tresh = 20

        final_epoch = start_epoch + self.cfg.TRAIN.NUM_EPOCHS
        for epoch in range(start_epoch, final_epoch):
            self.train(train_loader, optimizer, scheduler, train_meter, epoch=epoch)

            if (epoch + 1) % 2 == 0:
                val_meter.enable_confusion_mat()
                miou = self.eval(val_loader, val_meter, epoch=epoch)
                val_meter.disable_confusion_mat()

                if miou > best_miou:
                    best_miou = miou
                    best_epoch = epoch
                    logger.info(
                    f"New best checkpoint reached at epoch {epoch+1} with miou of {best_miou}\nSaving new best model."
                    )
                    cp.save_checkpoint(
                        self.checkpoint_dir,
                        epoch + 1,
                        best_miou,
                        self.cfg.NUM_GPUS,
                        self.cfg,
                        self.model,
                        optimizer,
                        scheduler,
                        best=True,
                    )

                elif epoch - best_epoch > early_stopping_tresh:
                    logger.info(f"Early stopped at trainning at epoch {epoch}")
                    break


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser("config file")

    parser.add_argument("--config_path", type=str, help="path of the cfg file to use")

    parser.add_argument("--train_path", type=str, help="path of the hdf5 file to use")

    parser.add_argument("--val_path", type=str, help="path of the hdf5 file to use")

    parser.add_argument("--num_channels", type=int, help="channels for the input")

    args = parser.parse_args()

    cfg = get_cfg_defaults()

    cfg.merge_from_file(args.config_path)

    cfg.DATA.PATH_HDF5_TRAIN = args.train_path

    cfg.DATA.PATH_HDF5_VAL = args.val_path

    cfg.MODEL.NUM_CHANNELS = args.num_channels

    Trainer = Trainer(cfg=cfg)

    Trainer.run()