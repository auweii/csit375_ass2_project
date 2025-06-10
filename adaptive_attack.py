import torch



def transform(
        x: torch.Tensor,
        exp_cfg,
) -> torch.Tensor:
    """
    x: normalized images of shape (N, C, H, W)
    exp_cfg: general configurations including out_dir, data_dir, device, etc.

    return: transformed images of shape (N, C, H, W)
    """

    raise NotImplementedError




