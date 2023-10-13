# -*- coding: utf-8 -*-

config = {
    "in_channels": 1,
    "out_channels": 2,
    "finalsigmoid": 1,
    "fmaps_degree": 16,
    "fmaps_layer_number": 4,
    "layer_order": "cip",
    "GroupNormNumber": 4,
    "device": "cuda:0",
    "weight_path": '../checkpoints/airway_model.pth',
    "roi_size": (128, 224, 304),
    "sw_batch_size": 1,
    "overlap": 0.50,
    "mode": 'constant',
    "sigma_scale": 0.125,
    "KeepLargestConnectedComponent": True,
    "use_HU_window":False
}
