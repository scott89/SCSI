from core.datasets.kitti import KITTI
from config.defult_config import config


kitti = KITTI(config.dataset.data_path, config.dataset.train_data_file, None, True, False, True)
print('Done!')
print(len(kitti.im_paths))
