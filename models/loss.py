# Author: Jintao Huang
# Date: 

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_focal_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none",
) -> torch.Tensor:
    """Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Paper: https://arxiv.org/abs/1708.02002.
    公式：FocalLoss = alpha * (1 - p_t) ^ gamma * ce_loss. CELoss = -log(pred) * target

    :param pred: shape = (N,). 未过sigmoid
    :param target: shape = (N,)
    :param alpha: float. Weighting factor in range (0,1) to balance. alpha = -1(< 0) (no weighting)
    :param gamma: float
    :param reduction: 'none' | 'mean' | 'sum'
    :return: shape = ()
    """
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p = torch.sigmoid(pred)
    p_t = target * p + (1 - target) * (1 - p)
    loss = ((1 - p_t) ** gamma) * ce_loss

    if alpha >= 0:
        alpha_t = target * alpha + (1 - target) * (1 - alpha)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class Loss(nn.Module):
    pass
