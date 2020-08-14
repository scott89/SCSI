import torch
import re
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
        self.color_jittering = T.ColorJitter()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    def __call__(self, data):
        augmentation = self.color_jittering.get_params(
            brightness=[max(0, 1 - self.brightness), 1 + self.brightness],
            contrast=[max(0, 1 - self.contrast), 1 + self.contrast],
            saturation=[max(0, 1 - self.saturation), 1 + self.saturation],
            hue=[-self.hue, self.hue])
        data['rgb'] = augmentation(data['rgb'])
        if 'rgb_context' in data:
            data['rgb_context'] = [augmentation(r) 
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
        if 'rgb_context' in data:
            data['rgb_context'] = [self.resize(r) 
                                   for r in data['rgb_context']]
        # resize depth
        if 'depth' in data:
            data['depth'] = cv2.resize(data['depth'], 
                                   tuple(self.image_shape[-1::-1]), interpolation=cv2.INTER_NEAREST)
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
        data['rgb_original'] = self.to_tensor(data['rgb_original']).type(self.dtype)
        if 'rgb_context' in data:
            data['rgb_context'] = [self.to_tensor(r).type(self.dtype) 
                                  for r in data['rgb_context']]
            data['rgb_context_original'] = [self.to_tensor(r).type(self.dtype) 
                                  for r in data['rgb_context_original']]
        #data['intrinsics'] = self.to_tensor(data['intrinsics'])
        data['intrinsics'] = torch.from_numpy(data['intrinsics'])
        if self.data_format == 'BGR':
            data['rgb'] = torch.flip(data['rgb'], dims=[0])
            #data['rgb_original'] = data['rgb_original'][-1::-1]
            if 'rgb_context' in data:
                data['rgb_context'] = [torch.flip(r, dims=[0]) for r in data['rgb_context']]
            #data['rgb_context_original'] = [r[-1::-1] for r in data['rgb_context_original']]

        if 'depth' in data.keys():
            data['depth'] = self.to_tensor(data['depth'])
        if 'pose' in data.keys():
            data['pose'] = self.to_tensor(data['pose'])
        return data

class Normalize(object):
    def __init__(self, mean, std):
        self.normalize = T.Normalize(mean, std, inplace=True)
    def __call__(self, data):
        data['rgb'] = self.normalize(data['rgb'])
        if 'rgb_context' in data:
            data['rgb_context'] = [self.normalize(r) for r in data['rgb_context']]
        return data

class Duplicate(object):
    def __init__(self, pre='rgb'):
        self.pre = pre
    def __call__(self, data):
        keys = [k for k in data if re.match(self.pre, k)]
        for k in keys:
            if isinstance(data[k], np.ndarray):
                data[k+'_original'] = data[k].copy()
            elif isinstance(data[k], list):
                data[k+'_original'] = [v.copy() for v in data[k]]
            else:
                try: 
                    data[k+'_original'] = data[k].copy()
                except:
                    raise ValueError('Failed to duplicate data %s'%k)
        return data


class Transform(object):
    def __init__(self, config):
        transforms = []
        if 'image_shape' in config.keys():
            transforms.append(Resize(config.image_shape))
        transforms.append(Duplicate())
        if 'jittering' in config.keys():
            transforms.append(ColorJittering(*config.jittering))
        transforms.append(ToTensor(config.format))
        transforms.append(Normalize(config.mean, config.std))
        self.transforms = T.Compose(transforms)
    def __call__(self, data):
        return self.transforms(data)
        



