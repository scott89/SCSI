from core.builders.build_dataset import build_dataset
from utils.view_synthesis import view_synthesis
import cv2
import numpy as np
from config.config import config


kitti = build_dataset(config, 'train')
for data in kitti:
    depth = data['depth']
    img = data['rgb_original']
    img_context = data['rgb_context_original'][0]
    pose = data['pose_context'][0]
    K = data['intrinsics']
    img_syn = view_synthesis(img, img_context, depth, pose, K)
    im = img.numpy()
    im_ct = img_context.numpy()
    im_syn = img_syn.numpy()
    for i in range(im.shape[0]): 
        cv2.imwrite('%d.png'%i, np.transpose(im[i,-1::-1], [1,2,0])*255) 
        cv2.imwrite('%d-syn.png'%i, np.transpose(im_syn[i,-1::-1], [1,2,0])*255)
        cv2.imwrite('%d-ctxt.png'%i, np.transpose(im_ct[i,-1::-1], [1,2,0])*255)




