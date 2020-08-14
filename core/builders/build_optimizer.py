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
    params = [
        {'params': disp_net.parameters(), 'lr': lr},
        {'params': pose_net.parameters(), 'lr': lr},
    ]
    optim = torch.optim.Adam(params=params, lr=lr)
    lr_lambda = lambda epoch: lr_decay_factor
    lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optim, lr_lambda)
    return optim, lr_scheduler

