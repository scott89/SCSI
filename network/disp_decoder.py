import torch
import torch.nn as nn
import torch.nn.functional as F

def disp_to_depth(disp, min_depth=0.1, max_depth=100):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, stride=1, dropout=0.0):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        if dropout > 0:
            self.conv2 = nn.Sequential(self.conv2, nn.Dropout2d(dropout))
        self.norm = norm_layer(out_channels)
        self.activ = nn.ReLU(inplace=True)
    def forward(self, x):
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_short = self.shortcut(x)
        out = self.activ(self.norm(x_out + x_short))
        return out

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, padding, stride=1, bias=True, norm_layer=None, activ=None):
        super(Conv2D, self).__init__()
        ops = []
        conv = nn.Conv2d(in_channels, out_channels, k_size, stride=stride, padding=padding, bias=bias) 
        ops.append(conv)
        if norm_layer is not None:
            norm = norm_layer(out_channels)
            ops.append(norm)
        if activ is not None:
            activ = activ(inplace=True)
            ops.append(activ)
        self.op = nn.Sequential(*ops)

    def forward(self, x):
        return self.op(x)

class Disp(nn.Module):
    def __init__(self, in_channels, out_channels=1, min_depth=0.5):
        super().__init__()
        self.min_depth = min_depth
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activ = nn.Sigmoid()
    def forward(self, x):
        pre = self.activ(self.conv1(x))
        pre = disp_to_depth(pre)[0]
        #pre = pre / self.min_depth
        return pre

class DispDecoder(nn.Module):
    def __init__(self, norm_layer=None):
        super(DispDecoder, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.dec5 = ResBlock(2048, 1024, norm_layer)
        self.dec4 = ResBlock(1024, 512, norm_layer)
        self.dec3 = ResBlock(512, 256, norm_layer)
        self.dec2 = ResBlock(256, 128, norm_layer)
        self.dec1 = ResBlock(128, 64, norm_layer)
        #
        self.iconv5 = Conv2D(2048, 1024, 3, 1, bias=False, norm_layer=norm_layer, activ=nn.ReLU)
        self.iconv4 = Conv2D(1024, 512, 3, 1, bias=False, norm_layer=norm_layer, activ=nn.ReLU)
        self.iconv3 = Conv2D(513, 256, 3, 1, bias=False, norm_layer=norm_layer, activ=nn.ReLU)
        self.iconv2 = Conv2D(129, 128, 3, 1, bias=False, norm_layer=norm_layer, activ=nn.ReLU)
        self.iconv1 = Conv2D(65, 64, 3, 1, bias=False, norm_layer=norm_layer, activ=nn.ReLU)
        #
        self.disp4 = Disp(512)
        self.disp3 = Disp(256)
        self.disp2 = Disp(128)
        self.disp1 = Disp(64)
    def forward(self, x):
        in1, in2, in3, in4, in5= x
        #
        dec5 = self.dec5(in5) # 1024
        dec5 = F.interpolate(dec5, in4.shape[2:], mode='bilinear', align_corners=False)
        con5 = torch.cat([dec5, in4], dim=1) # 2048
        iconv5 = self.iconv5(con5) # 1024
        #
        dec4 = self.dec4(iconv5) # 512
        dec4 = F.interpolate(dec4, in3.shape[2:], mode='bilinear', align_corners=False)
        con4 = torch.cat([dec4, in3], dim=1) # 1024
        iconv4 = self.iconv4(con4) # 512
        disp4 = self.disp4(iconv4)
        disp4_up = F.interpolate(disp4, in2.shape[2:], mode='bilinear', align_corners=False)
        #
        dec3 = self.dec3(iconv4) # 256
        dec3 = F.interpolate(dec3, in2.shape[2:], mode='bilinear', align_corners=False)
        con3 = torch.cat([dec3, in2, disp4_up], dim=1) # 513
        iconv3 = self.iconv3(con3) # 256
        disp3 = self.disp3(iconv3)
        disp3_up = F.interpolate(disp3, in1.shape[2:], mode='bilinear', align_corners=False)
        #
        dec2 = self.dec2(iconv3) # 128
        dec2 = F.interpolate(dec2, in1.shape[2:], mode='bilinear', align_corners=False)
        con2 = torch.cat([dec2, disp3_up], dim=1) # 129
        iconv2 = self.iconv2(con2) #128
        disp2 = self.disp2(iconv2)
        #disp2_up = F.interpolate(disp2, in1.shape[2:], mode='bilinear', align_corners=False)
        #
        dec1 = self.dec1(iconv2) # 64
        #dec1 = F.interpolate(dec1, in1.shape[2:], mode='bilinear', align_corners=False)
        con1 = torch.cat([dec1, disp2], dim=1) # 65
        iconv1 = self.iconv1(con1) # 64
        disp1 = self.disp1(iconv1)
        disp1 = F.interpolate(disp1, scale_factor=2, mode='bilinear', align_corners=False)
        return [disp4, disp3, disp2, disp1]


        



