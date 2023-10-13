# -*- coding: utf-8 -*-

from typing import Tuple, List, Dict, Union, Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.optim import lr_scheduler
import time

import sys
import os
import functools
import time
import numpy as np


def sample_gumbel(shape, device, eps=1e-20):
    U = torch.FloatTensor(shape).uniform_().to(device[0])
    return -(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, device, temperature, use_log=False):
    if use_log:
        y = torch.log(logits) + sample_gumbel(logits.size(), device)
    else:
        y = logits + sample_gumbel(logits.size(), device)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, device, temperature=0.1, use_log=True):
    y = gumbel_softmax_sample(logits, device, temperature, use_log)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


class LSD(nn.Module):
    def __init__(self, device, lse_kernel_size, lse_lamda, is_segmentation=True, **kwargs):
        super(LSD, self).__init__()
        self.device = device
        self.lse_lamda = lse_lamda
        self.lse_kernel_size = lse_kernel_size
        self.lse_kernel = torch.zeros(size=(1, 1, self.lse_kernel_size, self.lse_kernel_size, self.lse_kernel_size)).to(
            self.device[0])
        self.lse_kernel = self.init_LSE_kernel3d(self.lse_kernel, self.lse_lamda)

    def forward(self, x):
        # x is the likelihood map generated from the decoder.
        x_logits = torch.reshape(input=x, shape=(x_shape[1], -1)).transpose(1, 0)
        x_logits = torch.softmax(x_logits, dim=1)
        x_gumbel_softmax = gumbel_softmax(logits=x_logits, device=self.device, temperature=0.1, use_log=True)
        x_gumbel_softmax = x_gumbel_softmax.transpose(1, 0).reshape((x_shape))
        x_fg = x_gumbel_softmax[:, 1, ...].unsqueeze(1)
        img_dilated = self.soft_dilate(x_fg.detach())
        img_eroded = self.soft_erode(x_fg.detach())
        img_boundary = (img_dilated - img_eroded) * x_fg.detach()
        result = F.conv3d(img_boundary, self.lse_kernel, stride=1, padding=int((self.lse_kernel_size - 1) / 2))
        pred_cdt = (-self.lse_lamda * torch.log(result + 1e-10)) * x_fg.detach()
        return pred_cdt

    @staticmethod
    def soft_erode(img: torch.Tensor) -> torch.Tensor:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)

    @staticmethod
    def soft_dilate(img: torch.Tensor) -> torch.Tensor:
        p1 = F.max_pool3d(img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = F.max_pool3d(img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = F.max_pool3d(img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.max(torch.max(p1, p2), p3)

    @staticmethod
    def init_LSE_kernel3d(lse_kernel, lamda):
        kernel_size = lse_kernel.shape[2]
        kernel_centerpos = int((kernel_size - 1) / 2)
        lse_kernel[:, :, kernel_centerpos, kernel_centerpos, kernel_centerpos] = 0
        for i in range(0, kernel_size):
            for j in range(0, kernel_size):
                for k in range(0, kernel_size):
                    if (i == kernel_centerpos and j == kernel_centerpos and k == kernel_centerpos):
                        continue
                    else:
                        euclidean_distance = np.sqrt(
                            ((i - kernel_centerpos) ** 2 + (j - kernel_centerpos) ** 2 + (k - kernel_centerpos) ** 2))
                        if euclidean_distance > 50:
                            euclidean_distance = 50
                        euclidean_distance = torch.tensor(euclidean_distance)
                        lse_kernel[:, :, i, j, k] = torch.exp(- euclidean_distance / lamda)
        return lse_kernel
