from core.builders import *
from utils.misc import sample_to_cuda, model_restore

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
        for batch in train_dataloader:
            batch = sample_to_cuda(batch, config.model.gpu[0])
            disps = disp_net(batch['rgb'], flip_prob=0.5)
            poses = pose_net(batch['rgb'], batch['rgb_context'])


