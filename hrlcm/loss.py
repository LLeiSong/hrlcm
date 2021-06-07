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
import torch.nn.functional as F
from math import ceil
import numpy as np


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


def loss_coteaching_image(y_1, y_2, t, forget_rate):
    """loss for co-teaching, take the loss of each image as a whole
    This version works in semantic segmentation very similar to the original version,
    that takes each image as a whole thing. But it might be
    limited useful because it wastes many useful training labels.
    But good thing is it can filter some tiles with abnormal images.
    Original code: https://github.com/bhanML/Co-teaching/blob/master/loss.py
    """
    # Use balanced CE as the base loss function
    loss_fn = BalancedCrossEntropyLoss(reduction='none')
    loss_fn_update = BalancedCrossEntropyLoss()

    # Calculate the mean loss of a whole image and rank
    # Model1
    loss_ce_1 = loss_fn(y_1, t)
    loss_1 = torch.mean(loss_ce_1, dim=(1, 2))
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()

    # Model2
    loss_ce_2 = loss_fn(y_2, t)
    loss_2 = torch.mean(loss_ce_2, dim=(1, 2))
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()

    # Calc number of image within each mini-batch to remember
    remember_rate = 1 - forget_rate
    num_remember = ceil(remember_rate * len(ind_1_sorted))

    # Get the index of image within each mini-batch for loss
    ind_1_update = ind_1_sorted[:num_remember].cpu()
    ind_2_update = ind_2_sorted[:num_remember].cpu()

    # Calc updated loss of model1 and model2
    if len(ind_1_update) == 0:
        loss_1_update = loss_fn_update(y_1, t)
        loss_2_update = loss_fn_update(y_2, t)
    else:
        loss_1_update = loss_fn_update(y_1[ind_2_update], t[ind_2_update])
        loss_2_update = loss_fn_update(y_2[ind_1_update], t[ind_1_update])

    return loss_1_update, loss_2_update


def loss_colearning(y_1, y_2, t, forget_rate, noisy_or_not, mode='argue', golden_classes=None):
    """A combined version of co-teaching-plus loss and co-teaching loss.
        It takes the loss of each pixel separately in each image.
        We rename it loss_colearning because it is very different from the original ones already.
        Original code:
        https://github.com/bhanML/Co-teaching/blob/master/loss.py
        https://github.com/xingruiyu/coteaching_plus/blob/master/loss.py
    Args:
        y_1 (torch.Tensor): logits of model 1.
        y_2 (torch.Tensor): logits of model 2.
        t (torch.Tensor): labels.
        golden_classes: (list or None): Golden classes to consider, e.g. minority classes.
            It might not always helpful, so use it wisely.
        noisy_or_not: (list): a list of flags if the label is noisy or not.
            If it is a list of all True, then the loss becomes to
            a loss without any prior knowledge of label quality.
        mode (str): if use disagreement or not.
            argue is for using disagreement, discuss is for not. ['argue', 'discuss']
        forget_rate (float): the ratio of pixels to forget.
    returns:
        [torch.Tensor, torch.Tensor]: the loss of model 1 and model 2.
    """

    assert mode in ['argue', 'discuss']

    # Use balanced CE as the base loss function
    loss_fn = BalancedCrossEntropyLoss(reduction='none')
    loss_fn_update = BalancedCrossEntropyLoss()

    # Get prediction
    pred_1 = torch.max(F.softmax(y_1, dim=1), 1)[1]
    pred_2 = torch.max(F.softmax(y_2, dim=1), 1)[1]

    # Calculate loss of each pixel
    loss_ce_1 = loss_fn(y_1, t)
    loss_ce_2 = loss_fn(y_2, t)

    # Subset pixels within noisy image to calculate loss
    logits1_update = []
    target1_update = []
    logits2_update = []
    target2_update = []
    for idx, loss1 in enumerate(loss_ce_1):
        # Isolate this image
        pred1 = torch.flatten(pred_1[idx])
        pred2 = torch.flatten(pred_2[idx])
        y1 = torch.flatten(y_1[idx], start_dim=1)
        y2 = torch.flatten(y_2[idx], start_dim=1)
        loss1 = torch.flatten(loss1)
        loss2 = torch.flatten(loss_ce_2[idx])
        target = torch.flatten(t[idx])

        # If this image has noisy labels,
        # use disagreement and exchange small loss pixels
        if noisy_or_not[idx]:
            # Mask out disagree pixels
            if mode == 'argue':
                dis_mask = pred1 != pred2
                y1 = y1[:, dis_mask]
                y2 = y2[:, dis_mask]
                loss1 = loss1[dis_mask]
                loss2 = loss2[dis_mask]
                target = target[dis_mask]

            # Calc number of pixel within each image to remember
            remember_rate = 1 - forget_rate
            num_remember = int(remember_rate * len(loss1))

            # Select small loss pixels
            ind_1_sorted = np.argsort(loss1.cpu().data).cuda()
            ind_1_update = ind_1_sorted[:num_remember]
            ind_2_sorted = np.argsort(loss2.cpu().data).cuda()
            ind_2_update = ind_2_sorted[:num_remember]

            # If set golden classes (e.g. minority classes)
            if golden_classes is not None:
                ind_golden = []
                for val in golden_classes:
                    ind_golden.append((target == val).nonzero(as_tuple=False))
                ind_golden = torch.cat(ind_golden).squeeze()
                if ind_golden.dim() == 0:
                    ind_golden = ind_golden.unsqueeze(-1)

                ind_1_update = torch.cat([ind_golden, ind_1_update]).unique()
                ind_2_update = torch.cat([ind_golden, ind_2_update]).unique()

            # Get subset of each variables and append to the lists
            logits1_update.append(y1[:, ind_2_update])
            target1_update.append(target[ind_2_update])
            logits2_update.append(y2[:, ind_1_update])
            target2_update.append(target[ind_1_update])

        # If the image has clean labels, keep all pixels.
        else:
            logits1_update.append(y1)
            target1_update.append(target)
            logits2_update.append(y2)
            target2_update.append(target)

    logits1_update = torch.cat(logits1_update, 1)
    target1_update = torch.cat(target1_update)
    logits2_update = torch.cat(logits2_update, 1)
    target2_update = torch.cat(target2_update)

    # Calc updated loss of model1 and model2
    loss_1_update = loss_fn_update(logits1_update.unsqueeze(0), target1_update.unsqueeze(0))
    loss_2_update = loss_fn_update(logits2_update.unsqueeze(0), target2_update.unsqueeze(0))

    return loss_1_update, loss_2_update


def loss_colearning_batch(y_1, y_2, t, forget_rate, mode='argue', golden_classes=None):
    """A combined version of co-teaching-plus loss and co-teaching loss.
        It takes the loss of each pixel within the whole mini-batch.
        We rename it loss_colearning because it is very different from the original ones already.
        Original code:
        https://github.com/bhanML/Co-teaching/blob/master/loss.py
        https://github.com/xingruiyu/coteaching_plus/blob/master/loss.py
    Args:
        y_1 (torch.Tensor): logits of model 1.
        y_2 (torch.Tensor): logits of model 2.
        t (torch.Tensor): labels.
        golden_classes: (list or None): Golden classes to consider, e.g. minority classes.
            It might not always helpful, so use it wisely.
        mode (str): if use disagreement or not.
            argue is for using disagreement, discuss is for not. ['argue', 'discuss']
        forget_rate (float): the ratio of pixels to forget.
    returns:
        [torch.Tensor, torch.Tensor]: the loss of model 1 and model 2.
    """

    assert mode in ['argue', 'discuss']

    # Use balanced CE as the base loss function
    loss_fn = BalancedCrossEntropyLoss(reduction='none')
    loss_fn_update = BalancedCrossEntropyLoss()

    # Get prediction
    pred_1 = torch.max(F.softmax(y_1, dim=1), 1)[1]
    pred_2 = torch.max(F.softmax(y_2, dim=1), 1)[1]

    # Calculate loss of each pixel
    loss_ce_1 = loss_fn(y_1, t)
    loss_ce_2 = loss_fn(y_2, t)

    # Flatten the variables for this mini-batch
    pred_1 = torch.flatten(pred_1)
    pred_2 = torch.flatten(pred_2)
    loss_ce_1 = torch.flatten(loss_ce_1)
    loss_ce_2 = torch.flatten(loss_ce_2)
    y_1 = torch.flatten(y_1.permute(1, 0, 2, 3), start_dim=1)
    y_2 = torch.flatten(y_2.permute(1, 0, 2, 3), start_dim=1)
    t = torch.flatten(t)

    # If this image has noisy labels,
    # Mask out disagree pixels
    if mode == 'argue':
        dis_mask = pred_1 != pred_2
        y_1 = y_1[:, dis_mask]
        y_2 = y_2[:, dis_mask]
        loss_ce_1 = loss_ce_1[dis_mask]
        loss_ce_2 = loss_ce_2[dis_mask]
        t = t[dis_mask]

    # Calc number of pixel within each image to remember
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_ce_1))

    # Select small loss pixels
    ind_1_sorted = np.argsort(loss_ce_1.cpu().data).cuda()
    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_sorted = np.argsort(loss_ce_2.cpu().data).cuda()
    ind_2_update = ind_2_sorted[:num_remember]

    # If set golden classes (e.g. minority classes)
    if golden_classes is not None:
        ind_golden = []
        for val in golden_classes:
            ind_golden.append((t == val).nonzero(as_tuple=False))
        ind_golden = torch.cat(ind_golden).squeeze()
        if ind_golden.dim() == 0:
            ind_golden = ind_golden.unsqueeze(-1)

        ind_1_update = torch.cat([ind_golden, ind_1_update]).unique()
        ind_2_update = torch.cat([ind_golden, ind_2_update]).unique()

    logits1_update = y_1[:, ind_2_update]
    target1_update = t[ind_2_update]
    logits2_update = y_2[:, ind_1_update]
    target2_update = t[ind_1_update]

    # Calc updated loss of model1 and model2
    loss_1_update = loss_fn_update(logits1_update.unsqueeze(0), target1_update.unsqueeze(0))
    loss_2_update = loss_fn_update(logits2_update.unsqueeze(0), target2_update.unsqueeze(0))

    return loss_1_update, loss_2_update


def kl_loss_compute(pred, soft_targets, reduce=True):
    kl = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1), reduction='none')

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)


def loss_jocor(y_1, y_2, t, forget_rate, co_lambda=0.7, golden_classes=None):
    """A modified version of JoCoR loss.
        It takes the loss of each pixel within the whole mini-batch.
        Original code:
        https://github.com/hongxin001/JoCoR/blob/master/algorithm/loss.py
    Args:
        y_1 (torch.Tensor): logits of model 1.
        y_2 (torch.Tensor): logits of model 2.
        t (torch.Tensor): labels.
        co_lambda (float): the lambda value for co_jor.
        forget_rate (float): the ratio of pixels to forget.
        golden_classes: (list or None): Golden classes to consider, e.g. minority classes.
            It might not always helpful, so use it wisely.
    returns:
        torch.Tensor: the combined loss of model 1 and model 2.
    """
    loss_pick_1 = F.cross_entropy(y_1, t, reduction='none') * (1 - co_lambda)
    loss_pick_2 = F.cross_entropy(y_2, t, reduction='none') * (1 - co_lambda)
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y_1, y_2, reduce=False) +
                 co_lambda * kl_loss_compute(y_2, y_1, reduce=False)).flatten().cpu()

    ind_sorted = np.argsort(loss_pick.data).cuda()
    loss_sorted = loss_pick[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    # If set golden classes (e.g. minority classes)
    if golden_classes is not None:
        ind_golden = []
        for val in golden_classes:
            ind_golden.append((t == val).flatten().nonzero(as_tuple=False))
        ind_golden = torch.cat(ind_golden).squeeze()
        if ind_golden.dim() == 0:
            ind_golden = ind_golden.unsqueeze(-1)

        ind_update = torch.cat([ind_golden, ind_update]).unique()

    # Ensemble loss
    loss = torch.mean(loss_pick[ind_update])

    return loss


