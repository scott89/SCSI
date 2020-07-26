from easydict import EasyDict as edict
import numpy as np
import os

config = edict()
config.dataset = edict()
config.dataset.name = 'KITTI'
config.dataset.data_path = '/media/8TB/Research/Data/KITTI_raw'
config.dataset.train_data_file = 'data_splits/eigen_zhou_files.txt'
