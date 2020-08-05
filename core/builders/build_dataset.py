from core.builders.build_transform import build_transform
from core.datasets import *
from torch.utils.data import DataLoader

def worker_init_fn(worker_id):                                                                                                                                     
    """Function to initialize workers"""
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)


def build_dataset(config, phase):
    transform = build_transform(config, phase)
    data_path = config.dataset.data_path
    if phase == 'train':
        data_file = config.dataset.train_data_file
        with_context = True
        with_depth = True
        with_pose = True
        batch_size = config.dataset.train_batchsize
    elif phase == 'val':
        data_file = config.dataset.val_data_file
        with_context = False
        with_depth = True
        with_pose = False
        batch_size = config.dataset.val_batchsize

    dataset = eval(config.dataset.name)(data_path = data_path,
                                       data_file = data_file,
                                       data_transform = transform,
                                       with_context = with_context,
                                       with_depth = with_depth,
                                       with_pose = with_pose)
    dataset = DataLoader(dataset, 
                         batch_size = batch_size,
                         num_workers = config.dataset.num_workers,
                         pin_memory = True,
                         shuffle = phase=='train'
                        )
    return dataset
