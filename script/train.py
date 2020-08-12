import torch
import random
import numpy as np
from torch.backends import cudnn
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
    trainer(config)

