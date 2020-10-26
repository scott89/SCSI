from easydict import EasyDict as edict
import numpy as np
import os
config = edict()
config.dataset = edict()
config.dataset.name = 'KITTI'
config.dataset.data_path = '/media/8TB/Research/Data/KITTI_monodepth2'
config.dataset.num_workers = 12
config.dataset.data_file = 'data_splits/eigen_test_files.txt'
config.dataset.transform = edict()
config.dataset.transform.image_shape = [192, 640]
config.dataset.transform.keep_depth_size = True
config.dataset.transform.mean = [0.485, 0.456, 0.406]
config.dataset.transform.std = [0.229, 0.224, 0.225]
config.dataset.transform.format = 'RGB'

config.snapshot = 'models/res18-3d_v2.0.1/epoch-14.pth'
config.gpu = 1

