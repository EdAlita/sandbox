"""
models.py

This module defines deep learning models for 3D denoising tasks using PyTorch,
including UNet-based and competitive architecture variants. It supports model building
through configuration-driven design and modular encoder-decoder components.

Dependencies:
    - yacs.config: Handles model configuration via hierarchical config files.
    - torch: Provides tensor operations and neural network modules.
    - models.sub_models: Contains encoder, decoder, bottleneck, and auxiliary model blocks.
    - typing: Offers type hinting support for clarity and maintainability.

Owner:
    Edwing Ulin

Version:
    v1.0.1
"""
import yacs.config
import models.sub_models as sm
from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class FCDenseNet(nn.Module):
    def __init__(
        self,
        params: Dict,
    ):
        super().__init__()

        # Initial convolution
        self.first_conv = nn.Conv2d(params["num_channels"], 48, kernel_size=(params["kernel_h"], params["kernel_w"]), padding=1, bias=False)
        n_channels = 48

        # Encoder path
        self.dense_blocks_down = nn.ModuleList()
        self.trans_downs = nn.ModuleList()

        for n_layers in params["block_config"]:
            dense_block = sm.DenseBlock(n_channels, params["num_filters_interpol"], n_layers, params["drop_out_rate"])
            n_channels = dense_block.out_channels
            self.dense_blocks_down.append(dense_block)
            self.trans_downs.append(sm.TransitionDown(n_channels, params["drop_out_rate"]))

        # Bottleneck
        self.bottleneck = sm.DenseBlock(n_channels, params["num_filters_interpol"], params["num_filters"], params["drop_out_rate"])
        n_channels = self.bottleneck.out_channels

        # Decoder path
        self.trans_ups = nn.ModuleList()
        self.dense_blocks_up = nn.ModuleList()

        reversed_config = list(reversed(params["block_config"]))

        for i, n_layers in enumerate(reversed_config):
            self.trans_ups.append(sm.TransitionUp(n_channels, n_channels))
            n_channels = n_channels + self.dense_blocks_down[-(i + 1)].out_channels
            dense_block_up = sm.DenseBlock(n_channels, params["num_filters_interpol"], n_layers, params["drop_out_rate"])
            n_channels = dense_block_up.out_channels
            self.dense_blocks_up.append(dense_block_up)

        # Final convolution
        self.final_conv = nn.Conv2d(n_channels, params['n_classes'], kernel_size=(params["stride_conv"], params["stride_conv"]), padding=0, bias=False)

    def forward(self, x):
        out = self.first_conv(x)
        skip_connections = []

        # Encoder
        for dense_block, trans_down in zip(self.dense_blocks_down, self.trans_downs):
            out = dense_block(out)
            skip_connections.append(out)
            out = trans_down(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder
        for trans_up, dense_block_up in zip(self.trans_ups, self.dense_blocks_up):
            skip = skip_connections.pop()
            out = trans_up(out)

            # Shape alignment
            if out.size(2) != skip.size(2) or out.size(3) != skip.size(3):
                diffY = skip.size(2) - out.size(2)
                diffX = skip.size(3) - out.size(3)
                out = F.pad(out, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

            out = torch.cat([out, skip], dim=1)
            out = dense_block_up(out)

        out = self.final_conv(out)
        return out

#Dictionary of supported models
_MODELS = {
    "DenseNet": FCDenseNet
}


def build_model(cfg: yacs.config.CfgNode) -> Union[FCDenseNet]:
    """
    Build the requested model based on configuration.

    Parameters:
    ----------
    cfg: yacs.config.CfgNode
        Configuration node containing the model settings.

    Returns:
    -------
    model: UNETDIP
        The initialized UNETDIP model object.
    """
    # Assert if the model is supported
    assert (
        cfg.MODEL.MODEL_NAME in _MODELS.keys()
    ), f"Model {cfg.MODEL.MODEL_NAME} not supported"

    # Convert config keys to lowercase and initialize model
    params = {k.lower(): v for k, v in dict(cfg.MODEL).items()}
    model = _MODELS[cfg.MODEL.MODEL_NAME](params)

    return model
