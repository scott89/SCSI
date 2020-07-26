import os
from os.path import join
import torch
from torch.utils.data import Dataset
from PIL import Image
from config.defult_config import config



def _get_depth_path(im_path):
    depth_path = im_path.replace('image_02/data', 'proj_depth/velodyne/image_02')
    depth_path = depth_path.replace('png', 'npz')
    return depth_path

def _get_context_path(im_file, backward_context, forward_context, stride=1):
    parent_folder = os.path.dirname(im_file)
    base_name, ext = os.path.splitext(os.path.basename(im_file))
    c_id = int(base_name)
    # get backward context
    backward_context_paths = []
    for i in range(1, backward_context+1):
        b_id = c_id - i * stride
        b_path = join(parent_folder, str(b_id).zfill(len(base_name)) + ext)
        if os.path.exists(b_path):
            backward_context_paths.append(b_path)
        else:
            return None, None

    # get forward context
    forward_context_paths = []
    for i in range(1, backward_context+1):
        f_id = c_id + i * stride
        f_path = join(parent_folder, str(f_id).zfill(len(base_name)) + ext)
        if os.path.exists(f_path):
            forward_context_paths.append(f_path)
        else:
            return None, None

    return backward_context_paths, forward_context_paths

    


class KITTI(Dataset):
    def __init__(self, data_path, data_file, transform,
                with_context=True, with_depth=False, with_pose=False,
                backward_context=1, forward_context=1, stride=1):
        self.data_path = data_path
        self.with_context = with_context
        self.with_depth = with_depth
        self.with_pose = with_pose
        self.data_file = join(data_path, data_file)
        if with_context:
            self.backward_context_paths = []
            self.forward_context_paths = []
        with open(self.data_file, 'r') as fid:
            self.im_list = fid.readlines()
        im_paths = []
        if with_depth:
            depth_paths = []
        for imname in self.im_list:
            path = join(data_path, imname.split()[0])
            if not with_depth:
                im_paths.append(path)
            else:
                depth_path =  _get_depth_path(path)
                if depth_path is not None and os.path.exists(depth_path):
                    im_paths.append(path)
                    depth_paths.append(depth_path)

        if self.with_context:
            im_with_context_paths = []
            if with_depth:
                depth_with_context_paths = []
            depth_with_context_paths = []
            for idx, path in enumerate(im_paths):
                backward_context_path, forward_context_path = \
                        _get_context_path(path, backward_context, forward_context, stride)
                if backward_context_path is not None:
                    im_with_context_paths.append(path)
                    self.forward_context_paths.append(forward_context_path)
                    self.backward_context_paths.append(backward_context_path)
                    if with_depth:
                        depth_with_context_paths.append(depth_paths[idx])
            im_paths = im_with_context_paths
            if with_depth:
                depth_paths = depth_with_context_paths
        self.im_paths = im_paths
        if with_depth:
            self.depth_paths = depth_paths



