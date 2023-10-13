# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Tuple, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np
import collections
import enum


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


CUDA_ENABLED = True


def FloatTensor(device, *args):
    if CUDA_ENABLED:
        return torch.FloatTensor(*args).to(device)
    else:
        return torch.FloatTensor(*args)


def range_to_anchors_and_delta(precision_range, num_anchors, device):
    precision_values = np.linspace(
        start=precision_range[0], stop=precision_range[1], num=num_anchors + 1
    )[1:]

    delta = (precision_range[1] - precision_range[0]) / num_anchors
    return FloatTensor(device, precision_values), delta


def build_class_priors(
        labels,
        class_priors=None,
        weights=None,
        positive_pseudocount=1.0,
        negative_pseudocount=1.0,
):
    if class_priors is not None:
        return class_priors
    N, C = labels.size()
    weighted_label_counts = (weights * labels).sum(0)
    weight_sum = weights.sum(0)
    class_priors = torch.div(
        weighted_label_counts + positive_pseudocount,
        weight_sum + positive_pseudocount + negative_pseudocount
    )
    return class_priors


def weighted_hinge_loss(labels, logits, positive_weights=1.0, negative_weights=1.0):
    positive_term = (1 - logits).clamp(min=0) * labels
    negative_term = (1 + logits).clamp(min=0) * (1 - labels)
    if positive_weights_is_tensor and positive_term.dim() == 2:
        return (
                positive_term.unsqueeze(-1) * positive_weights
                + negative_term.unsqueeze(-1) * negative_weights
        )
    else:
        return positive_term * positive_weights + negative_term * negative_weights


def true_positives_lower_bound(labels, logits, weights):
    loss_on_positives = weighted_hinge_loss(labels, logits, negative_weights=0.0)
    weighted_loss_on_positives = (
        weights.unsqueeze(-1) * (labels - loss_on_positives)
        if loss_on_positives.dim() > weights.dim()
        else weights * (labels - loss_on_positives)
    )
    return weighted_loss_on_positives.sum(0)


def false_postives_upper_bound(labels, logits, weights):
    loss_on_negatives = weighted_hinge_loss(labels, logits, positive_weights=0)
    weighted_loss_on_negatives = (
        weights.unsqueeze(-1) * loss_on_negatives
        if loss_on_negatives.dim() > weights.dim()
        else weights * loss_on_negatives
    )
    return weighted_loss_on_negatives.sum(0)


class LagrangeMultiplier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def lagrange_multiplier(x):
    return LagrangeMultiplier.apply(x)
