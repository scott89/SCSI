from easydict import EasyDict as edict
import numpy as np
import os
config = edict()
config.dataset = edict()
config.dataset.name = 'KITTI'
config.dataset.data_path = '/media/8TB/Research/Data/KITTI_monodepth2'
config.dataset.num_workers = 12
file_path = 'data_splits/test'
data_file = os.listdir(os.path.join(config.dataset.data_path, file_path))
config.dataset.data_file = [os.path.join(file_path, d) for d in data_file]
config.dataset.transform = edict()
config.dataset.transform.image_shape = [192, 640]
config.dataset.transform.keep_depth_size = True
config.dataset.transform.mean = [0.485, 0.456, 0.406]
config.dataset.transform.std = [0.229, 0.224, 0.225]
config.dataset.transform.format = 'RGB'

config.snapshot = 'models/res18-3d_v2.1.3/epoch-23.pth'
config.gpu = 1

