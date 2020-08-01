import torch
import torch.nn as nn
import torch.nn.functional as F
from .disp_decoder import ResBlock


class PoseNet(nn.Module):
    def __init__(self, num_ref=2, norm_layer=None):
        super().__init__()
        self.num_ref = num_ref
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        nc = [16, 32, 64, 128, 256, 256, 256]
        self.res1=ResBlock((num_ref+1)*3, nc[0], norm_layer, stride=2)
        self.res2=ResBlock(nc[0], cn[1], norm_layer, stride=2)
        self.res3=ResBlock(nc[1], cn[2], norm_layer, stride=2)
        self.res4=ResBlock(nc[2], cn[3], norm_layer, stride=2)
        self.res5=ResBlock(nc[3], cn[4], norm_layer, stride=2)
        self.res6=ResBlock(nc[4], cn[5], norm_layer, stride=2)
        self.res7=ResBlock(nc[5], cn[6], norm_layer, stride=2)
        self.stem = nn.Sequential(self.res1, self.res2, self.res3, 
                                  self.res4, self.res5, self.res6, self.res7)
        self.pose_pred = nn.Conv2d(cn[6], 6*self.num_ref, 1)
    
    def forward(self, image, context):
        assert(len(context) == self.num_ref)
        x = torch.cat([image]+context, dim=1)
        stem = self.stem(x)
        pose = self.pose_pred(stem)
        pose = torch.mean(pose, [2,3])
        pose = 0.01 * pose.view(pose.size[0], -1, 6)
        return pose

        

