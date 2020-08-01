from core.build_dataset import build_dataset


kitti = build_dataset('train')
for d in kitti:
    a = d

