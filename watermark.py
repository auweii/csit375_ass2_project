import torch
import torch.nn as nn
import numpy as np


def train(
        ae: nn.Module,
        decoder: nn.Module,
        wm: np.array,
        dset: torch.utils.data.Dataset,
        exp_cfg,
) -> None:
    """
    ae: an autoencoder for generating watermarks that are added to clean images.
    decoder: the model for recovering watermarks from input.
    wm: the target watermark recovered by the decoder as type np.int64.
    dset: the dataset to train on.
    exp_cfg: general configurations including out_dir, data_dir, device, etc.

    return: None.
    """

    raise NotImplementedError

