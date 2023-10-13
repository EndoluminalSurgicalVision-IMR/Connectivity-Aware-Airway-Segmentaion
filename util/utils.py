# -*- coding: utf-8 -*-

import sys
import os

import SimpleITK as sitk
import numpy as np

from monai.inferers import sliding_window_inference
from monai.transforms import (
    KeepLargestConnectedComponent,
    ToNumpy,
    AsDiscrete,
    CastToType,
    AddChannel,
    SqueezeDim,
    ToTensor
)
import numpy as np
import torch


class InnerTransform(object):
    def __init__(self):
        self.ToNumpy = ToNumpy()
        self.ToNumpyFloat32 = ToNumpy(dtype=np.float32)
        self.AsDiscrete = AsDiscrete(threshold=0.5)
        self.ArgMax = AsDiscrete(argmax=True)
        self.KeepLargestConnectedComponent = KeepLargestConnectedComponent(applied_labels=1, connectivity=3)
        self.CastToNumpyUINT8 = CastToType(dtype=np.uint8)
        self.AddChannel = AddChannel()
        self.SqueezeDim = SqueezeDim()
        self.ToTensor = ToTensor(dtype=torch.float)


InnerTransformer = InnerTransform()


def save_itk(image, filename, origin, spacing, direction):
    if type(origin) != tuple:
        if type(origin) == list:
            origin = tuple(reversed(origin))
        else:
            origin = tuple(reversed(origin.tolist()))
    if type(spacing) != tuple:
        if type(spacing) == list:
            spacing = tuple(reversed(spacing))
        else:
            spacing = tuple(reversed(spacing.tolist()))
    if type(direction) != tuple:
        if type(direction) == list:
            direction = tuple(reversed(direction))
        else:
            direction = tuple(reversed(direction.tolist()))
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    itkimage.SetDirection(direction)
    sitk.WriteImage(itkimage, filename, True)


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = list(reversed(itkimage.GetOrigin()))
    numpySpacing = list(reversed(itkimage.GetSpacing()))
    numpyDirection = list(reversed(itkimage.GetDirection()))
    return numpyImage, numpyOrigin, numpySpacing, numpyDirection


def crop_image_via_box(image, box):
    return image[box[0, 0]:box[0, 1], box[1, 0]:box[1, 1], box[2, 0]:box[2, 1]]


def restore_image_via_box(origin_shape, image, box):
    origin_image = np.zeros(shape=origin_shape, dtype=np.uint8)  # np.uint8 is default
    origin_image[box[0, 0]:box[0, 1], box[1, 0]:box[1, 1], box[2, 0]:box[2, 1]] = image
    return origin_image

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)