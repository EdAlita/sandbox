from numbers import Number, Real
from typing import Union, Tuple, Any, Dict
import numpy as np
import numpy.typing as npt
import torch
from torchvision import transforms

class ToTensor_1input(object):
    def __call__(self, sample: npt.NDArray) -> Dict[str, Any]:

        img, label, weight = (
            sample['img'],
            sample['label'],
            sample['weight']
        )

        img = img.astype(np.float32)
        img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)
        img = img.transpose((2, 0, 1))

        return {
            "image": torch.from_numpy(img),
            "label": torch.from_numpy(label),
            "weight": torch.from_numpy(weight)
        }