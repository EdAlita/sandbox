"""
dataset.py

This module defines a PyTorch Dataset class (`Dataset3D`) for handling 3D image volumes,
including functionality for reading raw NIfTI files, adding Gaussian noise, and applying transformations.

Dependencies:
    - yacs.config: Handles configuration settings.
    - torch: Provides tensor operations and dataset utilities.
    - torch.utils.data.Dataset: Base class for creating custom datasets.
    - utils.misc: Contains helper functions for processing image records and reading NIfTI files.
    - utils.logger: Provides logging functionality.

Owner:
    Edwing Ulin

Version:
    v1.0.1
"""
import time
import h5py
from typing import List
from torch.utils.data import Dataset
from typing import Optional
import torch
import numpy as np
import yacs.config

from utils.gen_logger import loggen

logger = loggen(__name__)


class Dataset_V1(Dataset):
    """
    Class for loading aseg file for trainnig
    """

    def __init__(
            self,
            dataset_path: str,
            cfg: yacs.config.CfgNode,
            transforms: Optional = None): # type: ignore

        self.images = []
        self.labels = []
        self.subjects = []
        self.weights = []
        self.cfg = cfg
        self.transforms = transforms

        start = time.time()

        with h5py.File(dataset_path, "r") as hf:
            for size in hf.keys():
                try:
                    logger.info(f"Processing images of fold {size}.")
                    img_dataset = hf[f"{size}"]["volume"]
                    logger.info(
                        "Processed volumes of fold {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )

                    self.images.extend(img_dataset)

                    self.labels.extend(list(hf[f"{size}"]["gt"]))
                    logger.info(
                        "Processed segs of fold {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.subjects.extend(list(hf[f"{size}"]["subject"]))
                    logger.info(
                        "Processed subjects of fold {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.weights.extend(list(hf[f"{size}"]["weight"]))
                    logger.info(
                        "Processed weights of fold {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )

                except KeyError as e:
                    print(
                        f"KeyError: Unable to open object"
                    )
                    continue

            self.count = len(self.images)

            logger.info(
                "Successfully loaded {} data from {} in {:.3f} seconds".format(
                    self.count, dataset_path, time.time() - start
                )
            )

    def get_subject_names(self):
        return self.subjects

    def __getitem__(self, index):

        img = self.images[index]
        label = self.labels[index]
        weight = self.weights[index]

        tx_sample = self.transforms(
            {
                "img": img,
                "label": label,
                "weight": weight,
            }
            )
        return {
            "image": tx_sample["image"],
            "label": tx_sample["label"],
            "weight": tx_sample["weight"]
            }

    def __len__(self):
        """
        Get count.
        """
        return self.count


class Dataset_t1(Dataset):
    """
    Class for loading aseg file for trainnig
    """

    def __init__(
            self,
            dataset_path: str,
            cfg: yacs.config.CfgNode,
            transforms: Optional = None): # type: ignore

        self.images = []
        self.labels = []
        self.subjects = []
        self.weights = []
        self.cfg = cfg
        self.transforms = transforms
        start_global = time.time()
        start = time.time()

        with h5py.File(dataset_path, "r") as hf:
            for size in hf.keys():
                try:
                    logger.info(f"Processing images of fold {size}.")
                    img_dataset = hf[f"{size}"]["volume"]
                    logger.info(
                        "Processed volumes of fold {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )

                    self.images.extend(img_dataset)

                    self.labels.extend(list(hf[f"{size}"]["gt"]))
                    logger.info(
                        "Processed segs of fold {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.subjects.extend(list(hf[f"{size}"]["subject"]))
                    logger.info(
                        "Processed subjects of fold {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.weights.extend(list(hf[f"{size}"]["weight"]))
                    logger.info(
                        "Processed weights of fold {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )

                except KeyError as e:
                    print(
                        f"KeyError: Unable to open object (object {size} does not exist)"
                    )
                    continue

            self.count = len(self.images)

            logger.info(
                "Successfully loaded {} data from {} in {:.3f} seconds".format(
                    self.count, dataset_path, time.time() - start_global
                )
            )

    def get_subject_names(self):
        return self.subjects

    def __getitem__(self, index):

        img = self.images[index]
        label = self.labels[index]
        weight = self.weights[index]

        img = np.expand_dims(img.transpose((2, 0, 1)), axis=3)
        label = label[np.newaxis, :, :, np.newaxis]
        weight = weight[np.newaxis, :, :, np.newaxis]

        import torchio as tio

        subject = tio.Subject(
            {
                "img": tio.ScalarImage(tensor=img),
                "label": tio.LabelMap(tensor=label),
                "weight": tio.LabelMap(tensor=weight),
            }
        )

        if self.transforms is not None:
            tx_sample = self.transforms(subject)

        img = torch.squeeze(tx_sample["img"].data).float()
        label = torch.squeeze(tx_sample["label"].data).byte()
        weight = torch.squeeze(tx_sample["weight"].data).float()

        img = torch.clamp(img / 255.0, min=0.0, max=1.0)

        return {
            "image": img,
            "label": label,
            "weight": weight
        }

    def __len__(self):
        """
        Get count.
        """
        return self.count
