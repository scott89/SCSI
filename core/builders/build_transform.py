from core.datasets.transforms import Transform
def build_transform(config, phase='train'):
    
    if phase == 'train':
        trans_config = config.dataset.train_transform
    elif phase == 'val':
        trans_config = config.dataset.val_transform
    trans_config.image_shape = config.input.image_shape
    trans_config.mean = config.input.mean
    trans_config.std = config.input.std
    trans_config.format = config.input.format

    return Transform(trans_config)




