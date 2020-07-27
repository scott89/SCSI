import os
from os.path import join
import torch
from torch.utils.data import Dataset
from PIL import Image
from config.defult_config import config
import numpy as np


FOLDER = {'left': 'image_02', 
          'right':'image_04'}


class KITTI(Dataset):
    def __init__(self, data_path, data_file, transform,
                with_context=True, with_depth=False, with_pose=False,
                backward_context=1, forward_context=1, stride=1):
        self.data_path = data_path
        self.with_context = with_context
        self.with_depth = with_depth
        self.with_pose = with_pose
        self.data_file = join(data_path, data_file)
        self.calibration_cache = []
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
                depth_path =  self.get_depth_path(path)
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
                        self.get_context_path(path, backward_context, forward_context, stride)
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

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        data = {'image': Image.open(im_path)}
        # get intrinsics
        c_data = self.get_calibration(im_path)
        intrinsics = self.get_intrinsics(c_data, im_path)
        data.update({'intrinsics': intrinsics})
        # get pose
        if self.with_pose:
            data['pose'] = 

    
    @staticmethod
    def get_depth_path(im_path):
        for cam in FOLDER.values():
            if cam in im_path:
                depth_path = im_path.replace(cam+'/data', 'proj_depth/velodyne/'+cam)
                depth_path = depth_path.replace('png', 'npz')
                return depth_path
    
    @staticmethod
    def get_context_path(im_file, backward_context, forward_context, stride=1):
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

    def get_calibration(self, im_path):
        parent_path = os.path.dirname(im_path)+'../../../../'
        if parent_path in self.calibration_cache:
            c_data = self.calibration_cache[parent_path]
        else:
            c_data = self.read_calibration_file(parent_path)
        return c_data
    
    @staticmethod
    def read_calibration_file(parent_path):
        file_path = join(parent_path, 'b_cam_to_cam.txt')
        c_data = dict()
        with open(file_path, 'r') as fid:
            for line in fid.readlines():
                key, value = line.split(':', 1)
                try:
                    c_data[key] = np.array([float(i) for i in value.split()])
                except ValueError:
                    pass
        return c_data

    @staticmethod
    def get_intrinsics(c_data, im_path):
        for cam in FOLDER.values():
            if cam in im_path:
                intrinsics = c_data[cam.replace('Image', 'P_rect')]
                intrinsics = np.reshape(intrinsics, [3, 4])[:, :4]
                return intrinsics
     



