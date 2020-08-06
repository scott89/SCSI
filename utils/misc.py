import torch
import numpy as np


def sample_to_cuda(data, gpu_id=0, non_blocking=True):
    if isinstance(data, str):
        return data
    elif isinstance(data, dict):
        return {k: sample_to_cuda(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sample_to_cuda(d) for d in data]
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).pin_memory().to(gpu_id, non_blocking=True)
    elif isinstance(data, torch.Tensor):
        return data.pin_memory().to(gpu_id, non_blocking=True)
    else:
        raise ValueError('Unknown data type: %s'%(type(data)))

def model_restore(disp_net, pose_net, optim,
    resume, restore_optim, snapshot, backbone_path):
    gpu_id = torch.device(disp_net.device_ids[0])
    if resume:
        ckpt = torch.load(snapshot, map_location=gpu_id)
        disp_net.module.load_state_dict(ckpt['disp_net'])
        pose_net.module.load_state_dict(ckpt['pose_net'])
        if restore_optim:
            optim.load_state_dict(ckpt['optim'])
    else:
        ckpt = torch.load(backbone_path, map_location=gpu_id)
        disp_net.module.encoder.load_state_dict(ckpt)
