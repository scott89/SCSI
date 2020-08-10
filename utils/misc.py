import torch
import torch.nn.functional as F
import numpy as np


def sample_to_cuda(data, gpu_id=0, non_blocking=True):
    if isinstance(data, str):
        return data
    elif isinstance(data, dict):
        return {k: sample_to_cuda(v, gpu_id, non_blocking) for k, v in data.items()}
    elif isinstance(data, list):
        return [sample_to_cuda(d, gpu_id, non_blocking) for d in data]
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).pin_memory().to(gpu_id, non_blocking=True)
    elif isinstance(data, torch.Tensor):
        return data.pin_memory().to(gpu_id, non_blocking=True)
    else:
        raise ValueError('Unknown data type: %s'%(type(data)))

def model_restore(disp_net, pose_net, optim,
    resume, restore_optim, snapshot, backbone_path):
    gpu_id = torch.device(disp_net.device_ids[0])
    gpu_id = 'cpu'
    if resume:
        ckpt = torch.load(snapshot, map_location=gpu_id)
        disp_net.module.load_state_dict(ckpt['disp_net'])
        pose_net.module.load_state_dict(ckpt['pose_net'])
        if restore_optim:
            optim.load_state_dict(ckpt['optim'])
    else:
        ckpt = torch.load(backbone_path, map_location=gpu_id)
        disp_net.module.encoder.load_state_dict(ckpt, strict=False)

def disp2depth(disp):
    if isinstance(disp, list):
        return [disp2depth(d) for d in disp]
    else:
        return 1.0 / disp.clamp(min=1e-6)

def resize(x, shape, mode='bilinear', align_corners=False):
    if isinstance(x, list):
        return [resize(i, shape, mode, align_corners) for i in x]
    else:
        return F.interpolate(x, shape, mode=mode, align_corners=align_corners)

