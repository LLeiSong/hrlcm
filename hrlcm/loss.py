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


def weighted_loss(predict, target, weights=None, ignore_index=-100):
    """Train a single model

    :param predict: predicts
    :param target: validate
    :param weights: weights array
    :param ignore_index: the ignored index in labels
    :return: mean loss
    """
    # Calculate weighted mean loss
    if weights is None:
        loss_fn = BalancedCrossEntropyLoss(ignore_index=ignore_index)
        loss = loss_fn(predict, target)
    else:
        # Get class weights
        unique, unique_counts = torch.unique(target, return_counts=True)
        # Calculate weight for only valid indices
        unique_counts = unique_counts[unique != ignore_index]
        unique = unique[unique != ignore_index]
        ratio = unique_counts.float() / torch.numel(target)
        weight = (1. / ratio) / torch.sum(1. / ratio)

        lossWeight = torch.ones(predict.shape[1]).cuda() * 0.00001
        counts = torch.zeros(predict.shape[1]).cuda()
        for i in range(len(unique)):
            lossWeight[unique[i]] = weight[i]
            counts[unique[i]] = unique_counts[i]
            
        loss_fn = BalancedCrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        
        loss = loss_fn(predict, target)
        # Multiply image weights
        loss = torch.sum(loss, (1, 2)) * weights
        # Mean through batch
        loss = loss.sum() / (lossWeight * counts).sum()

    return loss
