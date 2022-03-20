"""
This is a script of multiple losses.
References:
    https://github.com/bermanmaxim/LovaszSoftmax
Author: Lei Song, Boka Luo
Maintainer: Lei Song (lsong@clarku.edu)
"""

from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

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
            if unique[i] == 0:
                lossWeight[unique[i]] = 2 * weight[i]
            else:
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


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore:  # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)]  # mean across images if per_image
    return 100 * np.array(ious)


def lovasz_softmax(predicts, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      predicts: [B, C, H, W] Variable, class modeled result at each prediction.
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    probas = F.softmax(predicts, dim=1)
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def lovasz_softmax_bce(predicts, labels, classes='present', per_image=False, ignore=None, alpha=0.5):
    bce = BalancedCrossEntropyLoss(ignore_index=ignore)
    if alpha == 1:
        loss = bce(predicts, labels)
    elif alpha == 0:
        loss = lovasz_softmax(predicts, labels, classes, per_image, ignore)
    else:
        loss = alpha * bce(predicts, labels) + (1 - alpha) * lovasz_softmax(predicts, labels,
                                                                            classes, per_image, ignore)
    return loss


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
