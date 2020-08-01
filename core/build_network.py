from config.config import config
from network.disp_net import DispNet
from network.pose_net import PoseNet
from torch.nn import DataParallel



def build_network():
    disp_net = DispNet(norm_layer=config.model.norm)
    disp_net = DataParallel(disp_net,device_ids=config.gpu).to(config.gpu[0])
    pose_net = PoseNet(norm_layer=config.model.norm)
    return disp_net, pose_net

