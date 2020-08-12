import torch
import numpy as np
import os
import random
from torch.backends import cudnn
from core.builders import *
from utils.misc import sample_to_cuda, model_restore, resize, write_train_summary_helper
from core.losses import calculate_loss
from core.validator import depth_validator

cudnn.benchmark = True
np.random.seed(128)
random.seed(128)
torch.cuda.manual_seed_all(128)
torch.manual_seed(128)

def trainer(config):
    disp_net, pose_net = build_network(config)
    train_dataloader = build_dataset(config, 'train')
    val_dataloader = build_dataset(config, 'val')
    optim, lr_scheduler = build_optimizer(config, disp_net, pose_net)
    train_summary, val_summary = build_summary_writer(config)
    start_epoch, start_step = model_restore(disp_net, pose_net, optim, 
                                            config.train.resume, config.train.restore_optim,
                                            config.train.snapshot, config.train.backbone_path)
    global_step = start_step
    for epoch in range(start_epoch, config.train.optim.max_epoch):
        disp_net.train()
        pose_net.train()
        for batch in train_dataloader:
            optim.zero_grad()
            batch = sample_to_cuda(batch, config.model.gpu[0])
            disps = disp_net(batch['rgb'], flip_prob=0.5)
            disps = resize(disps, shape=batch['rgb'].shape[2:], mode='bilinear')
            poses = pose_net(batch['rgb'], batch['rgb_context'])
            loss_all, loss = calculate_loss(batch['rgb_original'], batch['rgb_context_original'], 
                                                                    disps, poses, batch['intrinsics'], True)
            loss_all.backward()
            optim.step()
            if global_step % config.train.summary_step == 0 and global_step != start_step:
                write_train_summary_helper(train_summary, batch, disps, loss, global_step)
                
            if global_step % config.train.display_step == 0 and global_step != start_step:
                print("Iter: %d/%d, loss: %f, perc_loss: %f, ssim: %f, l1: %f, smooth_loss: %f"%
                      (global_step, len(train_dataloader), loss_all, loss['perc_loss'], loss['ssim_loss'], loss['l1_loss'], loss['smooth_loss']))
            global_step += 1

        if epoch % config.val.val_epoch == 0:
            depth_validator(disp_net,  val_dataloader, val_summary, epoch, global_step, config.model.gpu[0])
        if epoch in config.train.optim.lr_decay_epochs:
            lr_scheduler.step()
        if epoch % config.train.snapshot_epoch == 0:
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'disp_net': disp_net.module.state_dict(),
                'pose_net': pose_net.module.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'config': config
            }, os.path.join(config.train.output_path, 'epoch-%d.pth'%epoch))



