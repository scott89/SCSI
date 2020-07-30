from config.config import config
from core.datasets.transforms import Transform




def build_transform(phase='train'):
    if phase == 'train':
        return Transform(config.dataset.train_transform)
    elif phase == 'val':
        return Transform(config.dataset.val_transform)





