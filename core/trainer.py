from core.builders import *
from utils.misc import sample_to_cuda, model_restore, resize, write_train_summary_helper
from core.losses import calculate_loss

def trainer(config):
    disp_net, pose_net = build_network(config)
    train_dataloader = build_dataset(config, 'train')
    val_dataloader = build_dataset(config, 'val')
    optim, lr_scheduler = build_optimizer(config, disp_net, pose_net)
    train_summary, val_summary = build_summary_writer(config)
    model_restore(disp_net, pose_net, optim, 
                  config.train.resume, config.train.restore_optim,
                  config.train.snapshot, config.train.backbone_path)
    start_epoch = 0
    start_step = 0
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
            global_step += 1

            if global_step % config.train.summary_step == 0 and global_step != start_step:
                write_train_summary_helper(train_summary, batch, disps, loss, global_step)
                
            if global_step % config.train.display_step == 0 and global_step != start_step:
                print("Iter: %d, loss: %f, perc_loss: %f, ssim: %f, l1: %f, smooth_loss: %f"%
                      (global_step, loss_all, loss['perc_loss'], loss['ssim_loss'], loss['l1_loss'], loss['smooth_loss']))

        lr_scheduler.step()
        disp_net.eval()
        pose_net.eval()



