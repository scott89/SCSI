import torch
from torch import nn



def get_params(net, layer_type, include, ends, exclude=None, requires_grad=True):
    params = []
    param_names = []
    if not isinstance(include, list):
        include = [include]
    if not isinstance(layer_type, list):
        layer_type = [layer_type]
    if not isinstance(ends, list):
        ends = [ends]
    if isinstance(exclude, str):
        exclude = [exclude]

    for name, module in net.named_modules():
        if not type(module) in layer_type:
            continue
        for inc in include:
            if inc in name:
                exc_flag = True
                if exclude is not None:
                    for exc in exclude:
                        if exc in name:
                            exc_flag = False
                            break
                if exc_flag:
                    for pn, p in module.named_parameters():
                        for end in ends:
                            if pn.endswith(end):
                                if p.requires_grad == requires_grad:
                                    name = name+'.'+pn
                                    params.append(p)
                                    param_names.append(name)
                                break
                break
    return param_names, params

def build_optimizer(config, disp_net, pose_net):
    lr = config.train.optim.lr
    lr_decay_factor = config.train.optim.lr_decay_factor
    wd = config.train.optim.weight_decay
    #norm_layer = {'BN': nn.BatchNorm2d, 'GN': nn.GroupNorm}
    #norm_layer = norm_layer[config.model.norm]
    norm_layer = [nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm, nn.SyncBatchNorm]
    disp_norm = get_params(disp_net, norm_layer, '', ['weight', 'bias'])
    pose_norm = get_params(pose_net, norm_layer, '', ['weight', 'bias'])
    learnable_layers = [nn.Conv2d, nn.Linear]
    disp_weight = get_params(disp_net, learnable_layers, '', ['weight'])
    disp_bias = get_params(disp_net, learnable_layers, '', ['bias'])
    pose_weight = get_params(pose_net, learnable_layers, '', ['weight'])
    pose_bias = get_params(pose_net, learnable_layers, '', ['bias'])
    params = [
        {'params': disp_norm[1], 'lr': lr},
        {'params': pose_norm[1], 'lr': lr},
        {'params': disp_weight[1], 'lr': lr, 'weight_decay': wd},
        {'params': pose_weight[1], 'lr': lr},
        {'params': disp_bias[1], 'lr': lr},
        {'params': pose_bias[1], 'lr': lr}
    ]
    optim = torch.optim.Adam(params=params, lr=lr)
    lr_lambda = lambda epoch: lr_decay_factor
    lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optim, lr_lambda)
    return optim, lr_scheduler

