import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss_utils import *
from utils.view_synthesis import view_synthesis, compute_Kinv, project_2d3d, project_3d2d
from utils.misc import disp2depth
from pytorch3d import ops

def SSIM(x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
    """
    Structural SIMilarity (SSIM) distance between two images.

    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    C1,C2 : float
        SSIM parameters
    kernel_size,stride : int
        Convolutional parameters

    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM distance
    """
    pool2d = nn.AvgPool2d(kernel_size, stride=stride)
    refl = nn.ReflectionPad2d(1)

    x, y = refl(x), refl(y)
    mu_x = pool2d(x)
    mu_y = pool2d(y)

    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = pool2d(x.pow(2)) - mu_x_sq
    sigma_y = pool2d(y.pow(2)) - mu_y_sq
    sigma_xy = pool2d(x * y) - mu_x_mu_y
    v1 = 2 * sigma_xy + C2
    v2 = sigma_x + sigma_y + C2

    ssim_n = (2 * mu_x_mu_y + C1) * v1
    ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
    ssim = ssim_n / ssim_d
    ssim = torch.clamp((1 - ssim)/2, 0, 1)
    ssim = ssim.mean(1)
    return ssim

def l1_loss(x, y):
    l1 = torch.abs(x - y)
    l1 = l1.mean(1)
    return l1

def perceptual_loss(img, img_ref, depth, pose, K, Kinv, return_syn=False, alpha=0.85):
    '''
    img: Bx3xHxW
    img_ref: list of Bx3xHxW of length n
    disp: list of Bx3xhxw (different scales) of length m
    pose: Bxnx4x3
    Kinv: Bx3x3
    '''
    ssim = []
    l1 = []
    if return_syn:
        img_syns = []
    for d in depth:
        ssim_s = []
        l1_s = []
        if return_syn:
            img_syns_s = []
        for ic in range(len(img_ref)):
            img_syn = view_synthesis(img_ref[ic], d, pose[:, ic], K, Kinv)
            if return_syn:
                img_syns_s.append(img_syn)
            ssim_s.append(SSIM(img, img_syn))
            l1_s.append(l1_loss(img, img_syn))
            ssim_s.append(SSIM(img, img_ref[ic]))
            l1_s.append(l1_loss(img, img_ref[ic]))
        ssim.append(ssim_s)
        l1.append(l1_s)
        if return_syn:
            img_syns.append(img_syns_s)
    perc = [alpha * torch.stack(s, 1) + (1-alpha) * torch.stack(l, 1) for s, l in zip(ssim, l1)]
    perc = [torch.min(p, 1) for p in perc]
    perc, indics = zip(*perc)
    valid_masks = [[i == v*2 for i in indics] for v in range(2)]
    perc_loss = sum(torch.mean(p) for p in perc) / len(perc)
    loss = {'perc_loss': perc_loss}
    if return_syn:
        loss.update({'img_syns': img_syns})
    return loss, valid_masks


def smoothness_loss(disp, image, smooth_loss_weight):
    smoothness_x, smoothness_y = calc_smoothness(disp, image)
    smoothness = zip(smoothness_x, smoothness_y)
    smoothness = sum([(s_x.abs().mean() + s_y.abs().mean())/2.0**i for i, (s_x, s_y) in enumerate(smoothness)]) / len(disp)
    return smoothness * smooth_loss_weight

def compute_loss_3d(depth, depth_ref, scale, scale_ref, pose, K, Kinv, valid_mask, loss_weight, mode='bilinear', padding_mode='zeros', align_corners=True):
    num_scale = len(depth)
    num_view = len(depth_ref)
    B, _, H, W = depth[0].shape
    p3d = [[project_2d3d(d, Kinv, pose[:, v]) for d in depth] for v in range(num_view)]
    p3d_ref = [[project_2d3d(dr, Kinv) for dr in depth_ref[v]] for v in range(num_view)]
    p2d_target_ref = [[project_3d2d(p, K) for p in p3d[v]] for v in range(num_view)]
    p3d_ref_warp = []
    trans = []
    for p3d_view, p2d_t_r_view, p3d_r_view, mask_view in zip(p3d, p2d_target_ref, p3d_ref, valid_mask):
        p3d_ref_warp_view = []
        trans_view = []
        for p3d_scale, p2d_t_r_scale, p3d_r_scale, mask_scale in zip(p3d_view, p2d_t_r_view, p3d_r_view, mask_view):
            mask_scale *= (torch.abs(p2d_t_r_scale[...,0]) <= 1) * (torch.abs(p2d_t_r_scale[...,1]) <= 1) 
            p3d_ref_warp_scale = F.grid_sample(p3d_r_scale, p2d_t_r_scale, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
            t = ops.corresponding_points_alignment(p3d_scale.view(B,3,-1).permute([0,2,1]), 
                                                   p3d_ref_warp_scale.view(B, 3, -1).permute([0,2,1]), mask_scale.view(B, -1), estimate_scale=True)
            p3d_ref_warp_view.append(p3d_ref_warp_scale)
            trans_view.append(t)
        trans.append(trans_view)
    trans = list(zip(*trans))
    R = [torch.cat([t.R for t in t_s], 0) for t_s in trans] #[2B x 3 x 3] x num_scale
    T = [torch.cat([t.T for t in t_s], 0) for t_s in trans] #[2B x 3] x num_scale
    S = [torch.cat([t.s for t in t_s], 0) for t_s in trans] #[2B] x num_scale
    R_gt = torch.eye(3,3, device=depth[0].device).repeat([num_view*B, 1, 1]) 
    T_gt = torch.zeros_like(T[0])

    #S = torch.cat(S, 0).detach()# [2Bxnum_scale]
    S = torch.cat(S, 0).detach()# [2Bxnum_scale]

    #S_gt = torch.ones_like(S[0])
    scale = torch.squeeze(scale).repeat([num_view*num_scale])
    scale_ref = torch.squeeze(torch.cat(scale_ref, 0)).repeat([num_scale])

    R_loss = sum([torch.abs(r-R_gt).mean() for r in R]) / num_scale
    T_loss = sum([torch.abs(t-T_gt).mean() for t in T]) / num_scale
    #s_loss = torch.abs(scale*scale_ref.detach() - S*scale.clone()*scale_ref).mean()
    s_loss = torch.abs(scale*scale_ref.detach() - S*scale.detach()*scale_ref).mean()
    #s_loss = sum([torch.abs(s-S_gt).mean() for s in S]) / num_scale
    #s_loss = torch.abs(S*scale - scale_ref).mean()
    #
    loss_3d = R_loss + T_loss + 0.5*s_loss
    return loss_3d * loss_weight
    
            

def calculate_loss(img, img_context, disps, depths, scale,
                   depths_context, scale_context,
                   pose, K, return_syn=False, smooth_loss_weight=0.001, ssim_loss_weight=0.85, loss_3d_weight=0.01):
    Kinv = compute_Kinv(K)
    loss, valid_mask = perceptual_loss(img, img_context, depths, pose, K, Kinv, return_syn, ssim_loss_weight)
    loss_all = loss['perc_loss']
    if loss_3d_weight > 0:
        depths_context = [[d[0]] for d in depths_context]
        loss_3d = compute_loss_3d([depths[0]], depths_context, scale, scale_context, pose, K, Kinv, valid_mask, loss_3d_weight)
        loss_all += loss_3d
        loss.update({'loss_3d': loss_3d})
    if smooth_loss_weight > 0:
        smooth_loss = smoothness_loss(disps, img, smooth_loss_weight)
        loss_all += smooth_loss
        loss.update({'smooth_loss': smooth_loss})
    loss.update({'loss_all': loss_all})
    return loss_all, loss 




