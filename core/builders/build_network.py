from network.disp_net import DispNet
from network.pose_net import PoseNet
from torch.nn import DataParallel



def build_network(config):
    disp_net = DispNet(norm_layer=config.model.norm)
    disp_net = DataParallel(disp_net,device_ids=config.gpu).to(config.model.gpu[0])
    pose_net = PoseNet(norm_layer=config.model.norm)
    pose_net = DataParallel(pose_net,device_ids=config.gpu).to(config.model.gpu[0])
    return disp_net, pose_net

