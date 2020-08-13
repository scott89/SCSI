import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss_utils import *
from utils.view_synthesis import view_synthesis 
from utils.misc import disp2depth

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

def perceptual_loss(img, img_ref, disp, pose, K, return_syn=False, alpha=0.85):
    '''
    img: Bx3xHxW
    img_ref: list of Bx3xHxW of length n
    disp: list of Bx3xhxw (different scales) of length m
    pose: Bxnx4x3
    K: Bx3x3
    '''
    depth = disp2depth(disp)
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
            img_syn = view_synthesis(img_ref[ic], d, pose[:, ic], K)
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
    perc = [torch.min(p, 1)[0] for p in perc]
    perc_loss = sum(torch.mean(p) for p in perc) / len(perc)
    loss = {'perc_loss': perc_loss}
    if return_syn:
        loss.update({'img_syns': img_syns})
    return loss


def smoothness_loss(disp, image, smooth_loss_weight):
    smoothness_x, smoothness_y = calc_smoothness(disp, image)
    smoothness = zip(smoothness_x[-1::-1], smoothness_y[-1::-1])
    smoothness = sum([(s_x.abs().mean() + s_y.abs().mean())/2.0**i for i, (s_x, s_y) in enumerate(smoothness)]) / len(disp)
    return smoothness * smooth_loss_weight


def calculate_loss(img, img_ref, disp, pose, K, return_syn=False, smooth_loss_weight=0.001, ssim_loss_weight=0.85):
    loss = perceptual_loss(img, img_ref, disp, pose, K, return_syn, ssim_loss_weight)
    loss_all = loss['perc_loss']
    if smooth_loss_weight > 0:
        smooth_loss = smoothness_loss(disp, img, smooth_loss_weight)
        loss_all += smooth_loss
        loss.update({'smooth_loss': smooth_loss})
    loss.update({'loss_all': loss_all})
    return loss_all, loss 




