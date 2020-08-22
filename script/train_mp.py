import torch
import os
import random
import numpy as np
from torch.backends import cudnn
import torch.multiprocessing as mp
from core.trainer import trainer
from config.config import config


cudnn.benchmark = True
np.random.seed(128)
random.seed(128)
torch.cuda.manual_seed_all(128)
torch.manual_seed(128)

#import torch
#torch.autograd.set_detect_anomaly(True)



if __name__ == '__main__':
    os.environ['MASTER_ADDR']='127.0.0.1'
    os.environ['MASTER_PORT']='8887'
    ########################
    world_size = len(config.model.gpu)
    mp.spawn(trainer,
             args=(world_size, config),
             nprocs=world_size,
             join=True)

