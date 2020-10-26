import torch
import torch.nn.functional as F
import numpy as np


def sample_to_cuda(data, gpu_id=0, non_blocking=True):
    if isinstance(gpu_id, list):
        gpu_id = gpu_id[0]
    if isinstance(data, str):
        return data
    elif isinstance(data, dict):
        return {k: sample_to_cuda(v, gpu_id, non_blocking) for k, v in data.items()}
    elif isinstance(data, list):
        return [sample_to_cuda(d, gpu_id, non_blocking) for d in data]
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).pin_memory().to(gpu_id, non_blocking=True)
    elif isinstance(data, torch.Tensor):
        return data.pin_memory().to(gpu_id, non_blocking=True)
    else:
        raise ValueError('Unknown data type: %s'%(type(data)))

def model_restore(disp_net, pose_net, optim,
    resume, restore_optim, snapshot, backbone_path, rank, ddp, keep_lr=True):
    #gpu_id = torch.device(disp_net.device_ids[0])
    gpu_id = 'cpu'
    if resume:
        ckpt = torch.load(snapshot, map_location=gpu_id)
        disp_net.module.load_state_dict(ckpt['disp_net'])
        pose_net.module.load_state_dict(ckpt['pose_net'])
        if restore_optim:
            if keep_lr:
                lrs = []
                for g in optim.param_groups:
                    lrs.append(g['lr'])
            optim.load_state_dict(ckpt['optimizer_state_dict'])
            if keep_lr:
                for lr, g in zip(lrs, optim.param_groups):
                    g['lr'] = lr
        start_epoch = ckpt['epoch'] + 1
        start_step = ckpt['global_step']
    else:
        ckpt = torch.load(backbone_path, map_location=gpu_id)
     
        missing_keys, unexpected_keys = disp_net.module.encoder.encoder.load_state_dict(ckpt, strict=False)
        if rank == 0 or not ddp:
            print('Loading pretrained_params of disp_net ...')
            print('missing keys:')
            print(missing_keys)
            print('unexpected keys:')
            print(unexpected_keys)
        ckpt['conv1.weight'] = torch.cat([ckpt['conv1.weight']]*2, 1) / 2
        missing_keysm, unexpected_keys = pose_net.module.encoder.encoder.load_state_dict(ckpt, strict=False)
        if rank == 0 or not ddp:
            print('Loading pretrained_params of disp_net ...')
            print('missing keys:')
            print(missing_keys)
            print('unexpected keys:')
            print(unexpected_keys)
        start_epoch = 0
        start_step = 0
    return start_epoch, start_step

def disp2depth(disp):
    if isinstance(disp, list):
        return [disp2depth(d) for d in disp]
    else:
        return 1.0 / disp.clamp(min=1e-6)

def resize(x, shape, mode='bilinear', align_corners=False):
    if isinstance(x, list) or isinstance(x, tuple):
        return [resize(i, shape, mode, align_corners) for i in x]
    else:
        return F.interpolate(x, shape, mode=mode, align_corners=align_corners)

def norm(x):
    return (x - x.min())  / (x.max() - x.min())

def write_summary(writer, content, name, content_type, global_step, max_disp=2.0):
    content = content.detach().cpu().numpy()
    if content_type == 'scalar':
        writer.add_scalar(name, content, global_step=global_step)
    elif content_type == 'img':
        writer.add_image(name, content, global_step=global_step)
    elif content_type == 'disp':
        content = norm(content)
        #content = 255* (content / max_disp)
        writer.add_image(name, content, global_step=global_step)
    else:
        raise ValueError('Unknown content type: %s'%content_type)

def write_train_summary_helper(train_summary, batch, disps, loss, global_step):
    img_syns = loss['img_syns'][0]
    write_summary(train_summary, batch['rgb_original'][0], 'img', 'img', global_step)
    write_summary(train_summary, batch['rgb_context_original'][0][0], 'ref1', 'img', global_step)
    write_summary(train_summary, batch['rgb_context_original'][1][0], 'ref2', 'img', global_step)
    write_summary(train_summary, disps[0][0], 'disp', 'disp', global_step)
    write_summary(train_summary, img_syns[0][0], 'ref1_syn', 'img', global_step)
    write_summary(train_summary, img_syns[1][0], 'ref2_syn', 'img', global_step)
    write_summary(train_summary, loss['loss_all'], 'loss_all', 'scalar', global_step)
    write_summary(train_summary, loss['perc_loss'], 'perc_loss', 'scalar', global_step)
    write_summary(train_summary, loss['smooth_loss'], 'smooth_loss', 'scalar', global_step)

def write_val_summary_helper(val_summary, rgb, disp_gt, disp, metrics, metric_names, global_step):
    write_summary(val_summary, rgb[0], 'img', 'img', global_step)
    write_summary(val_summary, disp_gt[0], 'disp_gt', 'disp', global_step)
    write_summary(val_summary, disp[0], 'disp', 'disp', global_step)
    #metrics = metrics.cpu().numpy()
    for i, n in enumerate(metric_names):
        write_summary(val_summary, metrics[i], n, 'scalar', global_step)




