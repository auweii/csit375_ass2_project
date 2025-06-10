import numpy as np
import torch.nn as nn


def reverse_trigger(
        model: nn.Module,
        target_label: int,
        exp_cfg,
) -> (np.ndarray, np.ndarray):
    """
    model: the poisoned model.
    target_label: the target label of the backdoor attack.
    exp_cfg: general configurations including out_dir, data_dir, device, etc.

    return: the reversed trigger with shape [32, 32, 3] as np.uint8 and
            a trigger mask with shape [32, 32] as np.float32.
            Values in the trigger mask range from 0 to 1.

    Note that the trigger shape and target label are unknown in practice as adversaries will not share them.
    Nonetheless, we consider a simpler case in this assignment that the target label is known.
    """

    raise NotImplementedError



