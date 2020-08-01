import torch
import cv2
import random
import numpy as np
import torchvision.transforms as T
from PIL import Image


class ColorJittering(object):
    def __init__(self, brightness, contrast, saturation, hue):
        self.color_jittering = T.ColorJitter(brightness=brightness,
                                             contrast=contrast,
                                             saturation=saturation,
                                             hue=hue)
    def __call__(self, data):
        data['rgb'] = self.color_jittering(data['rgb'])
        if 'rgb_context' in data:
            data['rgb_context'] = [self.color_jittering(r) 
                                   for r in data['rgb_context']]
        return data

class Resize(object):
    def __init__(self, image_shape, interpolation=Image.ANTIALIAS):
        self.image_shape = image_shape
        self.interpolation = interpolation
        self.resize = T.Resize(image_shape, interpolation=interpolation)

    def __call__(self, data):
        w_ori, h_ori = data['rgb'].size
        h_out, w_out = self.image_shape
        # resize rgb
        data['rgb'] = self.resize(data['rgb'])
        data['rgb_context'] = [self.resize(r) 
                               for r in data['rgb_context']]
        # resize depth
        if 'depth' in data:
            data['depth'] = cv2.resize(data['depth'], 
                                   tuple(self.image_shape[-1::-1]), interpolation=cv2.INTER_LINEAR)
        # resize intrinsics
        intrinsics = np.copy(data['intrinsics'])                                                                                                                          
        intrinsics[0] *= w_out / w_ori
        intrinsics[1] *= h_out / h_ori
        data['intrinsics'] = intrinsics
        return data

class ToTensor(object):
    def __init__(self, data_format, dtype=torch.float32):
        assert data_format in ['RGB', 'BGR']
        self.data_format = data_format
        self.to_tensor = T.ToTensor()
        self.dtype = dtype
    def __call__(self, data):
        data['rgb'] = self.to_tensor(data['rgb']).type(self.dtype)
        data['rgb_context'] = [self.to_tensor(r).type(self.dtype) 
                              for r in data['rgb_context']]
        if self.data_format == 'BGR':
            data['rgb'] = data['rgb'][-1::-1]
            data['rgb_context'] = [r[-1::-1] for r in data['rgb_context']]

        if 'depth' in data.keys():
            data['depth'] = self.to_tensor(data['depth'])
        return data

class Normalize(object):
    def __init__(self, mean, std):
        self.normalize = T.Normalize(mean, std, inplace=True)
    def __call__(self, data):
        data['rgb'] = self.normalize(data['rgb'])
        data['rgb_context'] = [self.normalize(r) for r in data['rgb_context']]
        return data


class Transform(object):
    def __init__(self, config):
        transforms = []
        if 'jittering' in config.keys():
            transforms.append(ColorJittering(*config.jittering))
        if 'image_shape' in config.keys():
            transforms.append(Resize(config.image_shape))
        transforms.append(ToTensor(config.format))
        transforms.append(Normalize(config.mean, config.std))
        self.transforms = T.Compose(transforms)
    def __call__(self, data):
        return self.transforms(data)
        



