# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from .utils import get_tp_fp_fn, sum_tensor, Loss, lagrange_multiplier, build_class_priors, weighted_hinge_loss, FloatTensor, range_to_anchors_and_delta


class TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1., square=False, alpha=0.1, beta=0.9):
        super(TverskyLoss, self).__init__()
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()

        return -tversky


class CAS_COM(nn.Module):
    def __init__(self, apply_nonlin=None, alpha=0.1, beta=0.9, batch_dice=False, do_bg=True, smooth=1., square=False):
        super(CAS_COM, self).__init__()
        self.crossentropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.tversky = TverskyLoss(apply_nonlin=apply_nonlin, batch_dice=batch_dice, do_bg=do_bg, smooth=smooth,
                                   square=square,
                                   alpha=alpha, beta=beta)

    def forward(self, input, target, target_ce, weight_pixelmap, lamda1=1, lamda2=1):
        # weight_pixelmap is the distance transform map.
        y1 = self.tversky(input, target)
        y2 = torch.mean(torch.mul(self.crossentropy_loss(input, target_ce.long()), weight_pixelmap))
        CAS_com = y1 + y2
        return CAS_com


class CAS_COR(nn.Module):
    def __init__(self, device, weights=None, *args, **kwargs):
        super(CAS_COR, self).__init__()
        self.device = device
        self.precision_range_lower = precision_range_lower = 0.001
        self.precision_range_upper = precision_range_upper = 1.0
        self.num_classes = 2
        self.num_classes = 10

        self.precision_range = (
            self.precision_range_lower,
            self.precision_range_upper,
        )
        self.precision_values, self.delta = range_to_anchors_and_delta(
            self.precision_range, self.num_anchors, self.device
        )
        self.biases = nn.Parameter(
            FloatTensor(self.device, self.num_classes, self.num_anchors).zero_()
        )
        self.lambdas = nn.Parameter(
            FloatTensor(self.device, self.num_classes, self.num_anchors).data.fill_(
                1.0
            )
        )

    def forward(self, logits, targets, reduce=True, size_average=True, weights=None):
        logits = logits.view(-1, logits.shape[1])
        targets = targets.contiguous().view(-1)
        C = 1 if logits.dim() == 1 else logits.size(1)
        labels, weights = CAS_COR._prepare_labels_weights(
            logits, targets, device=self.device, weights=weights
        )
        lambdas = lagrange_multiplier(self.lambdas)
        hinge_loss = weighted_hinge_loss(
            labels.unsqueeze(-1),
            logits.unsqueeze(-1) - self.biases,
            positive_weights=1.0 + lambdas * (1.0 - self.precision_values),
            negative_weights=lambdas * self.precision_values
        )
        class_priors = loss_utils.build_class_priors(labels, weights=weights)
        lambda_term = class_priors.unsqueeze(-1) * (
                lambdas * (1.0 - self.precision_values)
        )
        per_anchor_loss = weights.unsqueeze(-1) * hinge_loss - lambda_term
        loss = per_anchor_loss.sum(2) * self.delta
        loss /= self.precision_range[1] - self.precision_range[0]
        if not reduce:
            return loss
        elif size_average:
            return loss.mean()
        else:
            return loss.sum()

    @staticmethod
    def _prepare_labels_weights(logits, targets, device, weights=None):
        N, C = logits.size()
        labels = FloatTensor(device, N, C).zero_().scatter(1, targets.unsqueeze(1).data, 1)
        if weights is None:
            weights = FloatTensor(device, N).data.fill_(1.0)
        if weights.dim() == 1:
            weights = weights.unsqueeze(-1)
        return labels, weights


class CAS(nn.Module):
    def __init__(self, device):
        super(CAS, self).__init__()
        self.COM = CAS_COM(apply_nonlin=torch.sigmoid)
        self.COR = CAS_COR(device)

    def forward(self, input, target, target_ce, weight_pixelmap, lamda1, lamda2):
        self.loss_1 = self.COM(input, target, target_ce, weight_pixelmap)
        self.loss_2 = self.COR(input, target_ce)
        return lamda1 * self.loss_1 + lamda2 * self.loss_2
