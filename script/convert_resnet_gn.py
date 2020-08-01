import torch
import torch.nn as nn
import pickle
from network.resnet import resnet50
import re

caffe2_model_path = './models/R-50-GN.pkl'
pytorch_model_path = './models/R-50-GN.pth'

# load pre-trained models
with open(caffe2_model_path, 'rb') as fid:
    caffe2_model = pickle.load(fid, encoding='latin1')
caffe2_model = caffe2_model['blobs']

# build pytorch resnet-gn model
norm_layer = lambda channels: nn.GroupNorm(32, channels)

resnet_gn = resnet50(norm_layer=norm_layer)

# mapping
gn_map = {'weight': 'gn_s', 'bias': 'gn_b'}
res_map = {'1': 'branch2a_', '2': 'branch2b_', '3': 'branch2c_'}
pre_map = {'fc.weight': 'pred_w', 'fc.bias': 'pred_b'}

# copy model weights
for n,m in resnet_gn.named_parameters():
    if re.match('conv1', n):
        key = 'conv1_w'
    elif re.match('bn1', n):
        key = 'conv1_'+gn_map[n.split('.')[1]]
    elif re.match('layer', n):
        n_list = n.split('.')
        key = 'res%d_%s_'%(int(n_list[0][5])+1, n_list[1])
        if n_list[2]=='downsample':
            key += 'branch1_'
            if n_list[3] == '0':
                key += 'w'
            else:
                key += gn_map[n_list[4]]
        else:
            key += res_map[n_list[2][-1]]
            if re.match('conv', n_list[2]):
                key += 'w'
            else:
                key += gn_map[n_list[3]]
    elif re.match('fc', n):
        key = pre_map[n]
    else:
        raise ValueError('Key not found: %s'%(n))
    print('%s \t\t %s'%(n, key))
    m.data.copy_(torch.tensor(caffe2_model[key]))

torch.save(resnet_gn.state_dict(), pytorch_model_path)

