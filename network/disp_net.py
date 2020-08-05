from network.resnet import resnet50
from network.disp_decoder import DispDecoder
import torch
import torch.nn as nn
import random
NORMS = {
    'BN': nn.BatchNorm2d,
    'GN': lambda num_channels: nn.GroupNorm(32, num_channels)
}

class DispNet(nn.Module):
    def __init__(self, norm_layer='BN'):
        super(DispNet, self).__init__()
        norm_layer = NORMS[norm_layer]
        self.encoder = resnet50(norm_layer=norm_layer)
        self.decoder = DispDecoder(norm_layer=norm_layer) 
    def forward(self, x, flip_prob=0.0):
        is_flip = random.random() < flip_prob
        if is_flip:
            x = torch.flip(x, [3])
        enc = self.encoder(x)
        disp = self.decoder(enc)
        if is_flip:
            disp = [torch.flip(d, [3]) for d in disp]
        return disp
