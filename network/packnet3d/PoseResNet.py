# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

#from packnet_sfm.networks.layers.resnet.resnet_encoder import ResnetEncoder
from network.packnet3d.resnet_encoder import ResnetEncoder
#from packnet_sfm.networks.layers.resnet.pose_decoder import PoseDecoder
from network.packnet3d.pose_decoder import PoseDecoder
from core.geometry.pose_utils import pose_vec2mat
from utils.misc import disp2depth
from pytorch3d import ops
from pytorch3d.ops.points_alignment import SimilarityTransform
from utils.view_synthesis import compute_Kinv
########################################################################################################################

class PoseResNet(nn.Module):
    """
    Pose network based on the ResNet architecture.

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
        assert version is not None, "PoseResNet needs a version"

        num_layers = int(version[:2])       # First two characters are the number of layers
        pretrained = version[2:] == 'pt'    # If the last characters are "pt", use ImageNet pretraining
        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)

        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained, num_input_images=2)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        self.scale = 2
        self.input_shape = [int(i/self.scale) for i in kwargs['input_shape']]
        self.B = kwargs['batch_size']
        self.device = kwargs['device'][0] if isinstance(kwargs['device'], list) else kwargs['device']
        self.avg_pool = nn.AvgPool2d(self.scale, self.scale)
        self.get_coords()

    def get_coords(self):
        ys, xs = torch.meshgrid(torch.arange(0, self.input_shape[0], device=self.device, dtype=torch.float32),
                                torch.arange(0, self.input_shape[1], device=self.device, dtype=torch.float32))
        zs = torch.ones(self.input_shape, dtype=torch.float32, device=self.device)
        coords = torch.stack([xs, ys, zs], 0) # 3 x H x W
        coords = coords.repeat([self.B, 1, 1, 1]) # B x 3 x H x W
        self.coords = coords.view([self.B, 3, -1]) # B x 3 x HW

    def forward(self, target_image, ref_imgs, target_disp, ref_disp, K, return_pose_vec=False):
        """
        Runs the network and returns predicted poses
        (1 for each reference image).
        """
        outputs = []
        for i, ref_img in enumerate(ref_imgs):
            inputs = torch.cat([target_image, ref_img], 1)
            axisangle, translation = self.decoder([self.encoder(inputs)])
            outputs.append(torch.cat([translation[:, 0], axisangle[:, 0]], 2))
        pose = torch.cat(outputs, 1)
        pose = pose.view(-1, 6)
        pose_mat = pose_vec2mat(pose)
        pose_mat = pose_mat.view(-1, 2, 3, 4)
        pose_mat = self.icp_refine(pose_mat, target_disp, ref_disp, K)
        if return_pose_vec:
            return pose_mat, pose.view([-1, 2, 6])
        return pose_mat
    def icp_refine(self, pose_mat, target_disp, ref_disp, K):
        target_depth = self.get_depth(target_disp[0]) # B x 1 x HW
        ref_depth = [self.get_depth(r[0]) for r in ref_disp]
        Kinv = self.get_Kinv(K)
        coords = self.coords[:Kinv.shape[0]]
        coord_3d = Kinv @ coords # B x 3 x HW
        target_3d = coord_3d * target_depth
        target_3d = target_3d.permute([0, 2, 1]) # B x HW x 3
        ref_3ds = [coord_3d * rd for rd in ref_depth]
        ref_3ds = [r.permute([0, 2, 1]) for r in ref_3ds]
        Rs = [pose_mat[:,i,:,:3].permute([0, 2, 1]) for i in range(pose_mat.shape[1])]
        Ts = [pose_mat[:,i,:,3] for i in range(pose_mat.shape[1])]
        init_poses = [SimilarityTransform(R=R, T=T, s=torch.ones([self.B], device=R.device, dtype=R.dtype)) for R, T in zip(Rs, Ts)]
        refine_poses = []
        for ref_3d, init_pose in zip(ref_3ds, init_poses):
            t = ops.iterative_closest_point(target_3d, ref_3d, init_transform=init_pose,
                                            estimate_scale=True, max_iterations=10)
            R = t.RTs.R.permute([0, 2, 1])
            T = t.RTs.T[:, :, None]
            refine_poses.append(torch.cat([R, T], 2))
        refine_poses = torch.stack(refine_poses, 1)
        return refine_poses
        


    def get_Kinv(self, K):
        K = K.clone()
        K[:, :2] /= self.scale
        Kinv = compute_Kinv(K)
        return Kinv

    def get_depth(self, disp):
        disp = self.avg_pool(disp)
        disp = disp2depth(disp)
        disp = disp.view(self.B, 1, -1)
        return disp

########################################################################################################################



