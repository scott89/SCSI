import torch
import torch.distributed as dist
import os
from core.builders import *
from utils.misc import sample_to_cuda, model_restore, resize, write_train_summary_helper
from core.losses import calculate_loss
from core.validator import depth_validator


def trainer(gpu_id, world_size, config, ddp=True):
    if ddp:
        dist.init_process_group("nccl", rank=gpu_id, world_size=world_size)
        torch.cuda.set_device(gpu_id)
    disp_net, pose_net = build_network(gpu_id, config, ddp)
    train_dataloader, train_sampler = build_dataset(config, 'train', gpu_id, world_size, ddp)
    optim, lr_scheduler = build_optimizer(config, disp_net, pose_net)
    start_epoch, start_step = model_restore(disp_net, pose_net, optim, 
                                            config.train.resume, config.train.restore_optim,
                                            config.train.snapshot, config.train.backbone_path,
                                            gpu_id, ddp, config.train.keep_lr)
    if gpu_id == 0 or not ddp:
        val_dataloader, _= build_dataset(config, 'val')
        train_summary, val_summary = build_summary_writer(config)
    global_step = start_step
    for epoch in range(start_epoch, config.train.optim.max_epoch):
        disp_net.train()
        pose_net.train()
        if ddp:
            train_sampler.set_epoch(epoch)
        for batch_id, batch in enumerate(train_dataloader):
            optim.zero_grad()
            batch = sample_to_cuda(batch, gpu_id)
            disps = disp_net(batch['rgb'], flip_prob=0.5)
            disps_context = [disp_net(c) for c in batch['rgb_context']]
            disps = resize(disps, shape=batch['rgb'].shape[2:], mode='bilinear')
            disps_context = resize(disps_context, shape=batch['rgb'].shape[2:], mode='bilinear')
            poses = pose_net(batch['rgb'], batch['rgb_context'], disps, disps_context, 
                            batch['intrinsics'])
            loss_all, loss = calculate_loss(batch['rgb_original'], batch['rgb_context_original'], 
                                                                    disps, disps_context, poses, batch['intrinsics'], True)
            loss_all.backward()
            optim.step()
            if global_step % config.train.summary_step == 0 and global_step != start_step and (gpu_id == 0 or not ddp):
                write_train_summary_helper(train_summary, batch, disps, loss, global_step)
                
            if global_step % config.train.display_step == 0 and global_step != start_step and (gpu_id == 0 or not ddp):
                print("Epoch: %d global_step: %d,  batch_id: %d/%d, loss: %f, perc_loss: %f, smooth_loss: %f"%
                      (epoch, global_step, batch_id, len(train_dataloader), loss_all, loss['perc_loss'],  loss['smooth_loss']))
            global_step += 1

        if epoch % config.val.val_epoch == 0 and (gpu_id == 0 or not ddp):
            depth_validator(disp_net,  val_dataloader, val_summary, epoch, global_step, gpu_id)
        if epoch in config.train.optim.lr_decay_epochs:
            lr_scheduler.step()
        if epoch % config.train.snapshot_epoch == 0 and (gpu_id == 0 or not ddp):
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'disp_net': disp_net.module.state_dict(),
                'pose_net': pose_net.module.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'config': config
            }, os.path.join(config.train.output_path, 'epoch-%d.pth'%epoch))



