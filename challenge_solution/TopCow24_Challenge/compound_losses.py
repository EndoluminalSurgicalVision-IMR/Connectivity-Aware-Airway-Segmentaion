import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss, SoftSkeletonRecallLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import numpy as np


class CE_SkelDistance_loss(nn.Module):
    def __init__(self):
        super(CE_SkelDistance_loss, self).__init__()
        self.ce_noreduce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, skel: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :param skel: Distance map weight, same size with the netoutput
        :return:
        """

        # def forward(self, input: Tensor, target: Tensor) -> Tensor:
        #     if target.ndim == input.ndim:
        #         assert target.shape[1] == 1
        #         target = target[:, 0]
        #     return super().forward(input, target.long())

        ce_skeldistance_loss = torch.mean(torch.mul(self.ce_noreduce(net_output, target[:, 0].long()), skel))

        return ce_skeldistance_loss


def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
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


class TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1., square=False, alpha=0.1, beta=0.9):
        """
        x : prediciton
        y : gt
        """
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


class CA_loss(nn.Module):
    def __init__(self, batch_dice, do_bg=True, alpha=0.1, beta=0.9, weight=1):
        super(CA_loss, self).__init__()
        self.ce_noreduce = nn.CrossEntropyLoss(reduction='none')
        self.weight = weight
        self.tversky = TverskyLoss(apply_nonlin=softmax_helper_dim1, batch_dice=batch_dice, do_bg=do_bg, alpha=alpha,
                                   beta=beta)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, skel: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :param skel: Distance map weight, same size with the netoutput
        :return:
        """

        ce_skeldistance_loss = torch.mean(torch.mul(self.ce_noreduce(net_output, target[:, 0].long()), skel))
        tversky_loss = self.tversky(net_output, target)
        total_loss = tversky_loss + self.weight * ce_skeldistance_loss

        return total_loss


class DC_CA_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, batch_dice, do_bg=True, alpha=0.1, beta=0.9, weight_ce=1, weight_dc=1,
                 weight_tv=1, ignore_label=None,dice_class=MemoryEfficientSoftDiceLoss):
        super(DC_CA_loss, self).__init__()

        self.ignore_label = ignore_label
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ce_noreduce = nn.CrossEntropyLoss(reduction='none')
        self.weight_ce = weight_ce
        self.weight_dc = weight_dc
        self.weight_tv = weight_tv
        self.tversky = TverskyLoss(apply_nonlin=softmax_helper_dim1, batch_dice=batch_dice, do_bg=do_bg, alpha=alpha,
                                   beta=beta)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, skel: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :param skel: Distance map weight, same size with the netoutput
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_CA_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dc != 0 else 0
        ce_skeldistance_loss = torch.mean(
            torch.mul(self.ce_noreduce(net_output, target[:, 0].long()), skel)) if self.weight_ce != 0 else 0
        tversky_loss = self.tversky(net_output, target) if self.weight_tv != 0 else 0
        total_loss = self.weight_dc * dc_loss + self.weight_tv * tversky_loss + self.weight_ce * ce_skeldistance_loss

        return total_loss



