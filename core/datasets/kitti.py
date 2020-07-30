import os
from os.path import join
import torch
from torch.utils.data import Dataset
from PIL import Image
from config.defult_config import config
import numpy as np
from core.datasets.kitti_utils import pose_from_oxts_packet, transform_from_rot_trans
from core.geometry.pose_utils import invert_pose_numpy


FOLDER = {'left': 'image_02', 
          'right':'image_03'}
CALIB_FILE = {
    'cam2cam': 'calib_cam_to_cam.txt',
    'velo2cam': 'calib_velo_to_cam.txt',
    'imu2velo': 'calib_imu_to_velo.txt',
}

class KITTI(Dataset):
    def __init__(self, data_path, data_file, data_transform=None,
                with_context=True, with_depth=False, with_pose=False,
                backward_context=1, forward_context=1, stride=1):
        self.data_path = data_path
        self.data_transform = data_transform
        self.with_context = with_context
        self.with_depth = with_depth
        self.with_pose = with_pose
        self.data_file = join(data_path, data_file)
        self.im_paths = []
        self.calibration_cache = dict()
        self.intrinsics_cache = dict()
        self.oxts_cache = dict()
        self.imu2velo_calib_cache = dict()
        if with_context:
            self.backward_context_paths = dict()
            self.forward_context_paths = dict()
        with open(self.data_file, 'r') as fid:
            im_list = fid.readlines()
        if with_depth:
            self.depth_paths = dict()
        if with_pose:
            self.pose_cache = dict()
        # load image path and corresponding depth paths
        for im_path in im_list:
            im_path = join(data_path, im_path.split()[0])
            if with_depth:
                depth_path =  self.get_depth_path(im_path)
                if depth_path is not None and os.path.exists(depth_path):
                    self.depth_paths[im_path] = depth_path
                else:
                    continue

            # load context  paths and filter out images without context
            if with_context:
                backward_context_path, forward_context_path = \
                        self.get_context_path(im_path, backward_context, forward_context, stride)
                if backward_context_path is not None:
                    self.forward_context_paths[im_path] = forward_context_path
                    self.backward_context_paths[im_path] = backward_context_path
                else:
                    continue
            # get_calibration, intrinsics and poses
            # calibration and intrinsics are automatically stored within member functions 
            c_data = self.get_calibration(im_path)
            intrinsics = self.get_intrinsics(c_data, im_path)
            # load pose
            if with_pose:
                self.get_pose(im_path)
            self.im_paths.append(im_path)


            

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        data = {'rgb': Image.open(im_path)}
        # get intrinsics
        intrinsics = self.intrinsics_cache[im_path]
        data.update({'intrinsics': intrinsics})
        # get pose
        if self.with_pose:
            data.update({'pose': self.pose_cache[im_path]})
        if self.with_depth:
            data.update({'depth': self.read_depth(self.depth_paths[im_path])})
        if self.with_context:
            context_paths = self.backward_context_paths[im_path] + \
                    self.forward_context_paths[im_path]
            image_context = [Image.open(f) for f in context_paths]
            data.update({'rgb_context': image_context})
            if self.with_pose:
                current_pose = data['pose']
                context_poses = [self.get_pose(f) for f in context_paths]
                context_poses = [invert_pose_numpy(p) @ current_pose
                                      for p in context_poses]
                data.update({'pose_context': context_poses})
        if self.data_transform is not None:
            data = self.data_transform(data)
        return data




    
    @staticmethod
    def read_depth(depth_path):
        depth = np.load(depth_path)['velodyne_depth'].astype(np.float32)
        return np.expand_dims(depth, axis=2)
                       
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
        parent_path = join(os.path.dirname(im_path), '../../../')
        if parent_path in self.calibration_cache:
            c_data = self.calibration_cache[parent_path]
            return c_data
        c_data = self.read_calibration_file(join(parent_path, CALIB_FILE['cam2cam']))
        self.calibration_cache[parent_path] = c_data
        return c_data
    
    @staticmethod
    def read_calibration_file(file_path):
        c_data = dict()
        with open(file_path, 'r') as fid:
            for line in fid.readlines():
                key, value = line.split(':', 1)
                try:
                    c_data[key] = np.array([float(i) for i in value.split()])
                except ValueError:
                    pass
        return c_data

    def get_intrinsics(self, c_data, im_path):
        if im_path in self.intrinsics_cache:
            return self.intrinsics_cache[im_path]
        for cam in FOLDER.values():
            if cam in im_path:
                intrinsics = c_data[cam.replace('image', 'P_rect')]
                intrinsics = np.reshape(intrinsics, [3, 4])[:, :3]
                self.intrinsics_cache[im_path] = intrinsics
                return intrinsics
     

    def get_pose(self, im_path):
        if im_path in self.pose_cache:
            return self.pose_cache[im_path]
        parent_path = os.path.dirname(im_path)
        base_name, ext = os.path.splitext(os.path.basename(im_path))
        origin_frame = join(parent_path, str(0).zfill(len(base_name)) + ext)
        # Get origin data
        origin_oxts_data = self.get_oxts_data(origin_frame)
        lat = origin_oxts_data[0]
        scale = np.cos(lat * np.pi / 180.)
        # Get origin pose
        origin_R, origin_t = pose_from_oxts_packet(origin_oxts_data, scale)
        origin_pose = transform_from_rot_trans(origin_R, origin_t)
        # Compute current pose
        oxts_data = self.get_oxts_data(im_path)
        R, t = pose_from_oxts_packet(oxts_data, scale)
        pose = transform_from_rot_trans(R, t)
        # Compute odometry pose
        imu2cam = self.get_imu2cam_transform(im_path, join(parent_path, '../../../'))
        odo_pose = (imu2cam @ np.linalg.inv(origin_pose) @
                    pose @ np.linalg.inv(imu2cam)).astype(np.float32)
        # Cache and return pose
        self.pose_cache[im_path] = odo_pose
        return odo_pose

    @staticmethod
    def get_oxts_file(image_file):
        """Gets the oxts file from an image file."""
        # find oxts pose file
        for cam in FOLDER.values():
            # Check for both cameras, if found replace and return file name
            if cam in image_file:
                return image_file.replace(cam, 'oxts').replace('.png', '.txt')
        # Something went wrong (invalid image file)
        raise ValueError('Invalid KITTI path for pose supervision.')

    def get_oxts_data(self, image_file):
        """Gets the oxts data from an image file."""
        oxts_file = self.get_oxts_file(image_file)
        if oxts_file in self.oxts_cache:
            oxts_data = self.oxts_cache[oxts_file]
        else:
            oxts_data = np.loadtxt(oxts_file, delimiter=' ', skiprows=0)
            self.oxts_cache[oxts_file] = oxts_data
        return oxts_data

        
    def get_imu2cam_transform(self, image_file, parent_folder):
        """Gets the transformation between IMU an camera from an image file"""
        #if image_file in self.imu2velo_calib_cache:
        #    return self.imu2velo_calib_cache[image_file]
        if parent_folder in self.imu2velo_calib_cache:
            return self.imu2velo_calib_cache[parent_folder]

        cam2cam = self.read_calibration_file(os.path.join(parent_folder, CALIB_FILE['cam2cam']))
        imu2velo = self.read_calibration_file(os.path.join(parent_folder, CALIB_FILE['imu2velo']))
        velo2cam = self.read_calibration_file(os.path.join(parent_folder, CALIB_FILE['velo2cam']))

        velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])
        imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])
        cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))

        imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat
        self.imu2velo_calib_cache[image_file] = imu2cam
        return imu2cam


