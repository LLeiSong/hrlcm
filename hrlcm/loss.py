"""
This is a script of multiple losses.
References:
    https://github.com/bermanmaxim/LovaszSoftmax
Author: Lei Song, Boka Luo
Maintainer: Lei Song (lsong@ucsb.edu)
"""

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from balanced_loss import Loss

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


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

    def __init__(self, reduction='mean'):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, predict, target):
        # get class weights
        unique, unique_counts = torch.unique(target, return_counts=True)
        ratio = unique_counts.float() / torch.numel(target)
        weight = (1. / ratio) / torch.sum(1. / ratio)

        lossWeight = torch.ones(predict.shape[1]).cuda() * 0.00001
        for i in range(len(unique)):
            lossWeight[unique[i]] = weight[i]
            # # Modify if you want to raise or reduce the weight of one class
            # if unique[i] == 0:
            #     lossWeight[unique[i]] = 2 * weight[i]
            # else:
            #     lossWeight[unique[i]] = weight[i]
        loss = nn.CrossEntropyLoss(weight=lossWeight, reduction=self.reduction)

        return loss(predict, target)
      
def HybridLoss(predict, target, ce_weight=0.5):
    # Potentially, predict always have results for num_class
    # But target may not have some classes
    unique, unique_counts = torch.unique(target, return_counts=True)
    unique_counts = unique_counts.detach().cpu().numpy().tolist()
    unique = unique.detach().cpu().numpy().tolist()
    nclass = predict.shape[1]
    
    # Fill the non-exist classes
    sample_num = [1] * nclass
    for i in range(len(unique)):
        sample_num[unique[i]] = unique_counts[i]
    
    # Transform the data for Loss
    predict = predict.permute(0,2,3,1)
    predict = torch.flatten(predict, end_dim=2)
    target = torch.flatten(target)
    
    # class-balanced focal loss
    focal_loss = Loss(
        loss_type="focal_loss",
        samples_per_class=sample_num,
        class_balanced=True)
    
    # class-balanced cross-entropy loss
    ce_loss = Loss(
        loss_type="cross_entropy",
        samples_per_class=sample_num,
        class_balanced=True)
    
    # Hybrid loss
    loss = ce_loss(predict, target) * ce_weight + focal_loss(predict, target) * (1 - ce_weight)
    
    return loss


# For U-Net 3plus with deep supervision only
def BalancedCrossEntropyLoss_3plus(predicts, target, reduction="mean", weights=[1.0, 0.25, 0.25, 0.25, 0.25]):
    loss_fn = BalancedCrossEntropyLoss(reduction=reduction)
    
    loss = (loss_fn(predicts[0], target) * weights[0] + loss_fn(predicts[1], target) * weights[1] + 
           loss_fn(predicts[2], target) * weights[2] + loss_fn(predicts[3], target) * weights[3] + 
           loss_fn(predicts[4], target) * weights[4]) / 2
    return loss


# For U-Net 3plus with deep supervision only
class LossMs(nn.Module):
    """
    Only used for deep supervision version of UNet3+
    Args:
        loss_fn (function): The loss function to calculate
        weights (list): the weights for loss calculated at different scales
    Returns:
        Loss tensor according to arg reduction
    """

    def __init__(self, loss_fn=None, weights=[1.0, 0.25, 0.25, 0.25, 0.25]):
        super(LossMs, self).__init__()
        self.loss_fn = loss_fn
        self.weights = weights

    def forward(self, predicts, target):
        loss = (self.loss_fn(predicts[0], target) * self.weights[0] +
                self.loss_fn(predicts[1], target) * self.weights[1] + 
                self.loss_fn(predicts[2], target) * self.weights[2] + 
                self.loss_fn(predicts[3], target) * self.weights[3] + 
                self.loss_fn(predicts[4], target) * self.weights[4]) / 2
        return loss
