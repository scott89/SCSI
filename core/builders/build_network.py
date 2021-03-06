#from network.disp_net import DispNet
#from network.pose_net import PoseNet
#from network.packnet3d.PackNet01 import PackNet01
from network.packnet3d.DepthResNet import DepthResNet
#from network.packnet3d.PoseNet import PoseNet
from network.packnet3d.PoseResNet import PoseResNet
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn


def build_network(gpu_id, config, ddp=True):
    disp_net = DepthResNet('18')
    pose_net = PoseResNet('18', 
                          input_shape=config.input.image_shape, 
                          batch_size=config.dataset.train_batchsize,
                          device=gpu_id)
    if config.model.norm == 'BN' and config.model.syn_norm and ddp:
        disp_net = nn.SyncBatchNorm.convert_sync_batchnorm(disp_net)
        pose_net = nn.SyncBatchNorm.convert_sync_batchnorm(pose_net)
    #disp_net = PackNet01()
    #disp_net = DepthResNet('50')
    if ddp:
        disp_net.to(gpu_id)
        pose_net.to(gpu_id)
        disp_net = DDP(disp_net, device_ids=[gpu_id])
        pose_net = DDP(pose_net, device_ids=[gpu_id])
    else:
        disp_net = DataParallel(disp_net,device_ids=config.model.gpu).to(config.model.gpu[0])
        pose_net = DataParallel(pose_net,device_ids=config.model.gpu).to(config.model.gpu[0])
    #pose_net = PoseNet(norm_layer=config.model.norm)
    #pose_net = PoseNet()
    #pose_net = DDP(pose_net, device_ids=[gpu_id], find_unused_parameters=True)
    #pose_net = DataParallel(pose_net,device_ids=config.model.gpu).to(config.model.gpu[0])
    return disp_net, pose_net

