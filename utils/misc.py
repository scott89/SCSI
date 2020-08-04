import torch
import numpy as np


def sample_to_cuda(data, gpu_id=0, non_blocking=True):
    if isinstance(data, str):
        return data
    elif isinstance(data, dict):
        return {k: sample_to_cuda(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [d for d in data]
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).pin_memory().to(gpu_id, non_blocking=True)
    elif isintance(data, torch.Tensor):
        return data.pin_memory().to(gpu_id, non_blocking=True)


