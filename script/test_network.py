from core.builders.build_network import build_network
from config.config import config
import torch
import torch.nn as nn
from core.builders.build_optimizer import get_params, build_optimizer

import torch.distributed as dist
import os
os.environ['MASTER_ADDR']='127.0.0.1'
os.environ['MASTER_PORT']='8887'

dist.init_process_group("nccl", rank=0, world_size=1)
disp_net, pose_net = build_network(0, config)
opt = build_optimizer(config, disp_net, pose_net)
n, p= get_params(disp_net, 'encoder', nn.Conv2d, 'weight')
pose_net = pose_net.module
pose_net.cpu()
model = torch.load('posenet.pth')
pose_net.load_state_dict(model)
