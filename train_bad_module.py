import torch
import torch.nn as nn


def train(
        bad_module: nn.Module,
        poison_target: int,
        trigger_size: int,
        trigger_alpha: float,
        exp_cfg,
) -> None:
    """
    bad_module: the module that detects the existence of triggers in input.
                The trigger is a red square at the bottom right corner of an image.
    poison_target: the target label of the attack.
    trigger_size: the size of the trigger.
    trigger_alpha: the alpha value of the trigger.
    exp_cfg: general configurations including out_dir, data_dir, device, etc.

    return: None.

    Note: you can only train this module on the TRAINING set of CIFAR10.
    If you train this module on the testing set of CIFAR10, marks will be deducted.
    This module will be evaluated on the testing set together with the target model.

    Hint: you may pass your own loss function to ModelTrainerConfig.
    """

    raise NotImplementedError


