from network.resnet import resnet50
from network.disp_decoder import DispDecoder
import torch.nn as nn
NORMS = {
    'BN': nn.BatchNorm2d,
    'GN': lambda num_channels: nn.GroupNorm(32, num_channels)
}

class DispNet(nn.Module):
    def __init__(self, norm_layer='BN'):
        super(ResBlock, self).__init__()
        norm_layer = NORMS[norm_layer]
        self.encoder = resnet50(norm_layer=norm_layer)
        self.decoder = DispDecoder(norm_layer=norm_layer) 
    def forward(self, x):
        enc = self.encoder(x)
        disp = self.decoder(enc)
        return disp
