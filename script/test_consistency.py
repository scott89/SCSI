from config.config import config
from torch.utils.data import DataLoader
from core.builders.build_transform import build_transform
from core.datasets import KITTI
from core.datasets.transforms import Transform
import os
import sys
from functools import partial
sys.path.append('/media/8TB/Research/Code/packnet-sfm')
from packnet_sfm.datasets.kitti_dataset import KITTIDataset
from packnet_sfm.datasets.transforms import train_transforms
from packnet_sfm.losses.multiview_photometric_loss import MultiViewPhotometricLoss
from packnet_sfm.geometry.pose import Pose
from core.builders.build_network import build_network
from utils.misc import sample_to_cuda, model_restore, resize, write_train_summary_helper
from core.losses import calculate_loss

data_path = config.dataset.data_path
data_file = config.dataset.train_data_file
with_context = True
with_depth = True
with_pose = False
batch_size = config.dataset.train_batchsize
transform = build_transform(config, 'train')
dataset = eval(config.dataset.name)(data_path = data_path,
                                   data_file = data_file,
                                   data_transform = transform,
                                   with_context = with_context,
                                   with_depth = with_depth,
                                   with_pose = with_pose) 
data_file = os.path.join(data_path, data_file)
transform = partial(train_transforms, image_shape=(192, 640), jittering=(0.2, 0.2, 0.2, 0.05))
kitti_dataset = KITTIDataset(data_path, data_file, data_transform=transform,
                            with_pose=with_pose, depth_type='velodyne', 
                             back_context=1, forward_context=1)
a = dataset[0]
b = kitti_dataset[0]

dataloader = DataLoader(dataset, 
                         batch_size = batch_size,
                         num_workers = 0,
                         pin_memory = True,
                         shuffle = False
                        )

data_it = dataloader.__iter__()
batch = data_it.next()
batch = sample_to_cuda(batch, config.model.gpu[0])
disp_net, pose_net = build_network(config)
disps = disp_net(batch['rgb'], flip_prob=0.5)
disps = resize(disps, shape=batch['rgb'].shape[2:], mode='bilinear')
pose_mat, poses = pose_net(batch['rgb'], batch['rgb_context'], True)
poses = [Pose.from_vec(poses[:, i], 'euler') 
         for i in range(poses.shape[1])]
disps = disps[-1:]
for i in range(100):
    loss_all, loss = calculate_loss(batch['rgb_original'], batch['rgb_context_original'], 
                                    disps, pose_mat, batch['intrinsics'], True)
    mvploss = MultiViewPhotometricLoss(num_scales=1,smooth_loss_weight=0.001, photometric_reduce_op='min',
                            automask_loss=True, clip_loss=0)
    loss2 = mvploss(batch['rgb_original'], batch['rgb_context_original'], disps[-1::-1], 
            batch['intrinsics'], batch['intrinsics'], poses)
