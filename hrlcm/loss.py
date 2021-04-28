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
from torch.autograd import Variable
import numpy as np


class BalancedCrossEntropyLoss(nn.Module):
    """
    Balanced cross entropy loss by weighting of inverse class ratio
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


class LabelSmoothSoftmaxCE(nn.Module):
    """
    Cross entropy with label smoothing
    Args:
        lb_smooth (float): the ratio to do label smoothing, should be (0, 1).
        reduction (str): Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
        ignore_index (int): Class index to ignore
    Returns:
        Loss tensor according to arg reduction
    """

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        """forward
        Args:
            logits: tensor of shape (N, C, H, W)
            label: tensor of shape(N, H, W)
        """
        # overcome ignored label
        logits = logits.float()  # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    """loss for co-teaching
    https://github.com/bhanML/Co-teaching/blob/master/loss.py
    """
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update = ind_1_sorted[:num_remember].cpu()
    ind_2_update = ind_2_sorted[:num_remember].cpu()
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()
        ind_2_update = ind_2_sorted.cpu().numpy()
        num_remember = ind_1_update.shape[0]

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_update]]) / float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_update]]) / float(num_remember)

    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update) / num_remember, torch.sum(loss_2_update) / num_remember, pure_ratio_1, pure_ratio_2


def loss_coteaching_plus(logits, logits2, labels, forget_rate, ind, noise_or_not, step):
    """loss-teaching-plus
    https://github.com/xingruiyu/coteaching_plus/blob/master/loss.py
    """
    outputs = F.softmax(logits, dim=1)
    outputs2 = F.softmax(logits2, dim=1)

    _, pred1 = torch.max(logits.data, 1)
    _, pred2 = torch.max(logits2.data, 1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    # TODO This part of disagree need to be updated
    # The idea could to get the disagree part within a tile
    # Then ignore the disagree part to calculate the loss
    # Meanwhile, an assumption could be if the disagree part exceed
    # a threshold, exchange idea.
    logical_disagree_id = np.zeros(labels.size(), dtype=bool)
    disagree_id = []
    for idx, p1 in enumerate(pred1):
        if p1 != pred2[idx]:
            disagree_id.append(idx)
            logical_disagree_id[idx] = True

    temp_disagree = ind * logical_disagree_id.astype(np.int64)
    ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
    try:
        assert ind_disagree.shape[0] == len(disagree_id)
    except:
        disagree_id = disagree_id[:ind_disagree.shape[0]]

    _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
    update_step = Variable(torch.from_numpy(_update_step)).cuda()

    if len(disagree_id) > 0:
        update_labels = labels[disagree_id]
        update_outputs = outputs[disagree_id]
        update_outputs2 = outputs2[disagree_id]

        loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(update_outputs, update_outputs2, update_labels,
                                                                     forget_rate, ind_disagree, noise_or_not)
    else:
        update_labels = labels
        update_outputs = outputs
        update_outputs2 = outputs2

        cross_entropy_1 = F.cross_entropy(update_outputs, update_labels)
        cross_entropy_2 = F.cross_entropy(update_outputs2, update_labels)

        loss_1 = torch.sum(update_step * cross_entropy_1) / labels.size()[0]
        loss_2 = torch.sum(update_step * cross_entropy_2) / labels.size()[0]

        pure_ratio_1 = np.sum(noise_or_not[ind]) / ind.shape[0]
        pure_ratio_2 = np.sum(noise_or_not[ind]) / ind.shape[0]
    return loss_1, loss_2, pure_ratio_1, pure_ratio_2


def kl_loss_compute(pred, soft_targets, reduce=True):
    kl = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1), reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)


def loss_jocor(y_1, y_2, t, forget_rate, ind, noise_or_not, co_lambda=0.1):
    loss_pick_1 = F.cross_entropy(y_1, t, reduce=False) * (1-co_lambda)
    loss_pick_2 = F.cross_entropy(y_2, t, reduce=False) * (1-co_lambda)
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y_1, y_2, reduce=False) +
                 co_lambda * kl_loss_compute(y_2, y_1, reduce=False)).cpu()

    ind_sorted = np.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    pure_ratio = np.sum(noise_or_not[ind[ind_sorted[:num_remember]]])/float(num_remember)

    ind_update = ind_sorted[:num_remember]

    # exchange
    loss = torch.mean(loss_pick[ind_update])

    return loss, loss, pure_ratio, pure_ratio

