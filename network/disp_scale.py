import torch
import torch.nn as nn
import torch.nn.functional as F
#from .packnet3d.depth_decoder import ResBlock


class ScaleDecoder(nn.Module):
    def __init__(self, in_channels):
        super(ScaleDecoder, self).__init__()
        self.decoder_ch = [in_channels, 256, 128, 1]
        self.deconv1 = nn.Conv2d(in_channels, 256, 1, bias=True)
        self.deconv2= nn.Conv2d(256, 256, 3, padding=1, bias=True)
        self.deconv3 = nn.Conv2d(256, 256, 3, padding=1, bias=True)
        self.deconv4 = nn.Conv2d(256, 1, 3, padding=1, bias=True)
        self.activ = nn.ELU(inplace=True)

    def forward(self, x):
        deconv1 = self.activ(self.deconv1(x))
        deconv2 = self.activ(self.deconv2(deconv1))
        deconv3 = self.activ(self.deconv3(deconv2))
        deconv4 = self.deconv4(deconv3)
        scale = 23 * deconv4.mean(3, keepdim=True).mean(2, keepdim=True)
        return scale
