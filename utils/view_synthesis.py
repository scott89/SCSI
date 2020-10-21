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



def project_2d3d(depth, Kinv, pose=None):
    '''
    input: 
        depth: B x 1 x H x W
        Kinv: B x 3 x 3
        pose: B x 3 x 4
    output:
        point_3d: B x 3 x H x W
    '''
    dtype = depth.dtype
    device = depth.device
    B, C, H, W = depth.shape
    #Kinv = compute_Kinv(K)
    ys, xs = torch.meshgrid(torch.arange(0, H, device=device, dtype=dtype),
                            torch.arange(0, W, device=device, dtype=dtype))
    zs = torch.ones([H, W], dtype=dtype, device=device)
    coords = torch.stack([xs, ys, zs], 0) # 3 x H x W
    coords = coords.repeat([B, 1, 1, 1]) # B x 3 x H x W
    coords = coords.view([B, 3, -1]) # B x 3 x HW
    points_3d = Kinv @ coords 
    points_3d = points_3d.view([B, 3, H, W])
    points_3d = points_3d * depth 
    if pose is not None:
        points_3d = pose[:, :3, :3] @ points_3d.view([B, 3, -1]) + pose[:, :3, 3:]
        points_3d = points_3d.view([B, 3, H, W])
    return points_3d

def project_3d2d(points_3d, K, pose=None):
    '''
    input: 
        point_3d: B x 3 x H x W
        K: B x 3 x 3
        pose: B x 3 x 4
    output:
        coords: B x H x W x 2: normalized to [-1,1]
    '''
    B, C, H, W = points_3d.shape
    points_3d = points_3d.view([B, C, -1])
    if pose is not None:
        points_3d = pose[:, :3, :3] @ points_3d + pose[:, :3, 3:]
        #points_3d = pose @ points_3d
    coords = K @ points_3d # B x 3 x HW
    xs = coords[:, 0]
    ys = coords[:, 1]
    zs = coords[:, 2].clamp(min=1e-5)
    x_norm = 2 * (xs / zs) / (W - 1) - 1 # B x HW
    y_norm = 2 * (ys / zs) / (H - 1) - 1
    coords = torch.stack([x_norm, y_norm], dim=2).view([B, H, W, 2])
    return coords

def view_synthesis(img_ref, depth, pose, K, Kinv, 
                   mode='bilinear', padding_mode='zeros', align_corners=True):
    '''
    img_ref: Bx3xHxW
    depth: Bx1xHxW
    pose: Bx3x4
    K: Bx3x3
    Kinv: Bx3x3
    '''
    world_points = project_2d3d(depth, Kinv)
    ref_coords = project_3d2d(world_points, K, pose)
    img_syn = F.grid_sample(img_ref, ref_coords, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return img_syn 
