# Copyright 2020 Toyota Research Institute.  All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from network.packnet3d.resnet_encoder import ResnetEncoder
from network.packnet3d.depth_decoder import DepthDecoder
from network.disp_decoder import disp_to_depth
from network.disp_scale import ScaleDecoder
import random

########################################################################################################################

def normalization(d, mode):
    if mode == 'mean':
        m = torch.mean(d, [2,3], keepdim=True)
    elif mode == 'median':
        m = torch.median(d.view([d.shape[0], -1]), 1, keepdim=True)
        m = m[..., None, None]
    else:
        raise ValueError('Unknown normalization mode: %s'%mode)
    return d / m.clamp(1e-6)


class DepthResNet(nn.Module):
    """
    Inverse depth network based on the ResNet architecture.

    Parameters
    ----------
    version : str
        Has a XY format, where:
        X is the number of residual layers [18, 34, 50] and
        Y is an optional ImageNet pretrained flag added by the "pt" suffix
        Example: "18pt" initializes a pretrained ResNet18, and "34" initializes a ResNet34 from scratch
    kwargs : dict
        Extra parameters
    """
    def __init__(self, version=None, **kwargs):
        super().__init__()
        assert version is not None, "DispResNet needs a version"

        num_layers = int(version[:2])       # First two characters are the number of layers
        pretrained = version[2:] == 'pt'    # If the last characters are "pt", use ImageNet pretraining
        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)

        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        self.scale_decoder = ScaleDecoder(self.encoder.num_ch_enc[-1])
        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.5, max_depth=100.0)

    def forward(self, x, flip_prob=0.0):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        #is_flip = random.random() < flip_prob
        #if is_flip:
        #    x = torch.flip(x, [3])

        fea = self.encoder(x)
        x = self.decoder(fea)
        disps = [x[('disp', i)] for i in range(4)]
        scale = self.scale_decoder(fea[-1])
        disps = [self.scale_inv_depth(d)[0] for d in disps]
        depth_norm = [normalization(1.0/d, mode='mean') for d in disps]
        depth = [scale * d for d in depth_norm]
        #if is_flip:
        #    disps = [torch.flip(d, [3]) for d in disps]
        return disps, depth, scale
########################################################################################################################
