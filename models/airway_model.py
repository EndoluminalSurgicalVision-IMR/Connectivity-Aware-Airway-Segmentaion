# -*- coding: utf-8 -*-


from networks.airway_network import UNet3D, normalize_CT, lumTrans
from configs.airway_config import config
from util.utils import InnerTransformer

import torch
import numpy as np


class AirwayExtractionModel(object):
    def __init__(self):
        self.config = config
        self.device = []
        self.device.append(self.config['device'])
        self.net = UNet3D(
            in_channels=self.config['in_channels'],
            out_channels=self.config['out_channels'],
            finalsigmoid=self.config['finalsigmoid'],
            fmaps_degree=self.config['fmaps_degree'],
            fmaps_layer_number=self.config['fmaps_layer_number'],
            layer_order=self.config['layer_order'],
            GroupNormNumber=self.config['GroupNormNumber'],
            device=self.device
        )

        self.net.load_state_dict(
            torch.load(self.config['weight_path'], map_location=lambda storage, loc: storage.cuda(0)))

    @torch.no_grad()
    def predict(self, image: np.ndarray):
        self.net.eval()
        if self.config['use_HU_window']:
            image = lumTrans(image)
        image = normalize_CT(image)
        image = InnerTransformer.ToTensor(image)
        image = InnerTransformer.AddChannel(image)
        image = InnerTransformer.AddChannel(image)
        image = image.to(self.device[0])

        pred = sliding_window_inference(
            inputs=image,
            roi_size=self.config['roi_size'],
            sw_batch_size=self.config['sw_batch_size'],
            predictor=self.net,
            overlap=self.config['overlap'],
            mode=self.config['mode'],
            sigma_scale=self.config['sigma_scale']
        )

        pred = InnerTransformer.AsDiscrete(pred[:, 1, ...])
        if self.config['KeepLargestConnectedComponent']:
            pred = InnerTransformer.KeepLargestConnectedComponent(pred)
        pred = InnerTransformer.ToNumpy(pred)
        pred = InnerTransformer.CastToNumpyUINT8(pred[0, ...])
        torch.cuda.empty_cache()
        return pred
