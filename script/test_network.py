from core.builders.build_network import build_network
from config.config import config
import torch
import torch.nn as nn
from core.builders.build_optimizer import get_parames

disp_net, pose_net = build_network(config)
n, p= get_parames(disp_net, 'encoder', nn.Conv2d, 'weight')
pose_net = pose_net.module
pose_net.cpu()
model = torch.load('posenet.pth')
pose_net.load_state_dict(model)
