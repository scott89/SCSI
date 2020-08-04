import torch
import torch.nn.functional as F


def compute_Kinv(K):
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]
    Kinv = K.clone()
    Kinv[:, 0, 0] = 1 / fx
    Kinv[:, 1, 1] = 1 / fy
    Kinv[:, 0, 2] = -cx / fx
    Kinv[:, 1, 2] = -cy / fy
    return Kinv



def project_2d3d(depth, K, pose=None):
    dtype = depth.dtype
    device = depth.device
    B, C, H, W = depth.shape
    Kinv = compute_Kinv(K)
    ys, xs = torch.meshgrid(torch.arange(0, H, device=device, dtype=dtype),
                            torch.arange(0, H, device=device, dtype=dtype))
    zs = torch.ones([H, W], dtype=dtype, device=device)
    coords = torch.stack([xs, ys, zs], 0) # 3 x H x W
    coords = coords.repeat([B, 1, 1, 1]) # B x 3 x H x W
    coords = coords.view([B, 3, -1]) # B x 3 x HW
    3d_points = Kinv @ coords 
    3d_points = 3d_points.view([B, 3, H, W])
    3d_points = 3d_points * depth 
    if pose is not None:
        3d_points = pose @ 3d_points
    return 3d_points

def project_3d2d(3d_points, K, pose=None):
    B, C, H, W = 3d_points.shape
    if pose is not None:
        3d_points = pose @ 3d_points
    coords = K @ 3d_points # B x 3 x H x W
    xs = coords[:, 0]
    ys = coords[:, 1]
    zs = coords[:, 2].clamp(min=1e-5)
    x_norm = 2 * (xs / zs) / (W - 1) - 1 # B x H x W
    y_norm = 2 * (ys / zs) / (H - 1) - 1
    coords = torch.stack([x_norm, y_norm], dim=1)
    return coords

def view_synthesis(img, img_ref, depth, pose, K, 
                   mode='bilinear', padding_mode='zeros', align_corners=True):
    world_points = project_2d3d(depth, K)
    ref_coords = project_3d2d(word_points, K, pose)
    img_syn = F.grid_sample(img_ref, ref_coords, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return img_syn 
