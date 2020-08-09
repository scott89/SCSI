import torch
import torch.nn as nn
import torch.nn.functional as F
from .disp_decoder import ResBlock
from core.geometry.pose_utils import pose_vec2mat

NORMS = {
    'BN': nn.BatchNorm2d,
    'GN': lambda num_channels: nn.GroupNorm(16, num_channels)
}

class PoseNet(nn.Module):
    def __init__(self, num_ref=2, norm_layer='BN'):
        super().__init__()
        self.num_ref = num_ref
        norm_layer = NORMS[norm_layer]
        nc = [32, 32, 64, 128, 256, 256, 256]
        self.res1=ResBlock((num_ref+1)*3, nc[0], norm_layer, stride=2)
        self.res2=ResBlock(nc[0], nc[1], norm_layer, stride=2)
        self.res3=ResBlock(nc[1], nc[2], norm_layer, stride=2)
        self.res4=ResBlock(nc[2], nc[3], norm_layer, stride=2)
        self.res5=ResBlock(nc[3], nc[4], norm_layer, stride=2)
        self.res6=ResBlock(nc[4], nc[5], norm_layer, stride=2)
        self.res7=ResBlock(nc[5], nc[6], norm_layer, stride=2)
        self.stem = nn.Sequential(self.res1, self.res2, self.res3, 
                                  self.res4, self.res5, self.res6, self.res7)
        self.pose_pred = nn.Conv2d(nc[6], 6*self.num_ref, 1)
    
    def forward(self, image, context):
        assert(len(context) == self.num_ref)
        x = torch.cat([image]+context, dim=1)
        stem = self.stem(x)
        pose = self.pose_pred(stem)
        pose = torch.mean(pose, [2,3])
        #pose = 0.01 * pose.view(pose.shape[0], -1, 6)
        pose = 0.01 * pose.view(-1, 6)
        pose_mat = pose_vec2mat(pose)
        pose_mat = pose_mat.view(-1, self.num_ref, 3, 4)
        return pose_mat

        

