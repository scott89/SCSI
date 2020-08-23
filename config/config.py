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
config.dataset.train_transform.jittering = [0.2, 0.2, 0.2, 0.05]
config.dataset.train_batchsize = 16
config.dataset.val_data_file = 'data_splits/eigen_test_files.txt'
config.dataset.val_transform = edict()
config.dataset.val_batchsize = 1

# Model
config.model = edict()
config.model.norm = 'BN'
config.model.syn_norm = True
config.model.gpu = [2, 5]

gpu_str = ','.join(map(str, config.model.gpu))
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
config.model.gpu = list(range(len(config.model.gpu)))

# train
config.train = edict()
config.train.resume = False
config.train.restore_optim = False  
config.train.snapshot = ''
config.train.output_path = 'models/baseline_packnet_mp_v1.0'
config.train.display_step = 50
config.train.summary_step = 200
config.train.snapshot_epoch = 1
if config.model.norm == 'BN':
    config.train.backbone_path = 'models/resnet50.pth'
elif config.model.norm == 'GN':
    config.train.backbone_path = 'models/R-50-GN.pth'
else:
    raise ValueError('Not Implemented %s'%config.model.norm)
config.train.optim = edict()
config.train.optim.lr = 4*1e-4
config.train.optim.weight_decay = 1e-4
config.train.optim.momentum = 0.9
config.train.optim.lr_decay_factor = 0.5
config.train.optim.lr_decay_epochs = [1000, 2000]
config.train.optim.max_epoch = 1000

#
config.val = edict()
config.val.val_epoch = 1

# Input
config.input = edict()
config.input.image_shape = [192, 640]
if config.model.norm == 'BN':
    # for ResNet with BN
    config.input.mean = [0.485, 0.456, 0.406]
    config.input.std = [0.229, 0.224, 0.225]
    config.input.format = 'RGB' 
elif config.model.norm == 'GN':
    # for ResNet with GN
    config.input.mean = [103.530/255, 116.280/255, 123.675/255]
    config.input.std = [1.0/255, 1.0/255, 1.0/255]
    config.input.format = 'BGR' 
else:
    raise ValueError('Not Implemented %s'%config.model.norm)
