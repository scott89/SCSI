from easydict import EasyDict as edict
import numpy as np
import os

config = edict()
config.dataset = edict()
config.dataset.name = 'KITTI'
config.dataset.data_path = '/media/8TB/Research/Data/KITTI_raw'
config.dataset.num_workers = 8
config.dataset.train_data_file = 'data_splits/eigen_zhou_files.txt'
config.dataset.train_transform = edict()
config.dataset.train_transform.image_shape = [192, 640]
config.dataset.train_transform.jittering = [0.2, 0.2, 0.2, 0.05]
config.dataset.train_transform.mean = [0.485, 0.456, 0.406]
config.dataset.train_transform.std = [0.229, 0.224, 0.225]
config.dataset.train_batchsize = 8
config.dataset.val_transform = edict()
config.dataset.val_transform.image_shape = [192, 640]
config.dataset.val_transform.mean = [0.485, 0.456, 0.406]
config.dataset.val_transform.std = [0.229, 0.224, 0.225]
config.dataset.val_batchsize = 1
