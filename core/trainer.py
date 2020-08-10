from core.builders import *
from utils.misc import sample_to_cuda, model_restore, resize
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
    for epoch in range(config.train.optim.max_epoch):
        disp_net.train()
        for batch in train_dataloader:
            optim.zero_grad()
            batch = sample_to_cuda(batch, config.model.gpu[0])
            disps = disp_net(batch['rgb'], flip_prob=0.5)
            disps = resize(disps, shape=batch['rgb'].shape[2:], mode='bilinear')
            poses = pose_net(batch['rgb'], batch['rgb_context'])
            loss, perc_loss, smooth_loss = calculate_loss(batch['rgb_original'], batch['rgb_context_original'], disps, poses, batch['intrinsics'])
            loss.backward()
            optim.step()



