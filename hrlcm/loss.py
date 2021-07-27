"""
This is a script of multiple losses.
References:
    https://github.com/bhanML/Co-teaching.git
    https://github.com/xingruiyu/coteaching_plus.git
    https://github.com/hongxin001/JoCoR.git
Author: Lei Song, Boka Luo
Maintainer: Lei Song (lsong@clarku.edu)
"""

from __future__ import print_function
import torch
import torch.nn as nn


class BalancedCrossEntropyLoss(nn.Module):
    """
    Balanced cross entropy loss by weighting of inverse class ratio
    Author: Boka Luo
    Args:
        ignore_index (int): Class index to ignore
        reduction (str): Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    """

    def __init__(self, ignore_index=-100, reduction='mean'):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, predict, target):
        # get class weights
        unique, unique_counts = torch.unique(target, return_counts=True)
        # calculate weight for only valid indices
        unique_counts = unique_counts[unique != self.ignore_index]
        unique = unique[unique != self.ignore_index]
        ratio = unique_counts.float() / torch.numel(target)
        weight = (1. / ratio) / torch.sum(1. / ratio)

        lossWeight = torch.ones(predict.shape[1]).cuda() * 0.00001
        for i in range(len(unique)):
            lossWeight[unique[i]] = weight[i]
        loss = nn.CrossEntropyLoss(weight=lossWeight, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss(predict, target)


def weighted_loss(predict, target, weights=None):
    """Train a single model

    :param predict: predicts
    :param target: validate
    :param weights: weights array
    :return: mean loss
    """
    # Calculate weighted mean loss
    loss_fn = BalancedCrossEntropyLoss()
    if weights is None:
        loss = loss_fn(predict, target)
    else:
        loss_total = 0
        for i in range(len(weights)):
            # An arbitrary way to set dims
            loss_each = loss_fn(torch.unsqueeze(predict[i, :, :, :], 0),
                                torch.unsqueeze(target[i, :, :], 0))
            loss_each = loss_each * weights[i]
            loss_total += loss_each
        loss = loss_total / len(weights)

    return loss
