from core.builders import *


def trainer(config):
    disp_net, pose_net = build_network(config)
    train_dataloader = build_dataset(config, 'train')
    val_dataloader = build_dataset(config, 'val')
    optim, lr_scheduler = build_optimizer(config, disp_net, pose_net)
    train_summary, val_summary = build_summary_writter(config)
    model_restore(disp_net, pose_net, optim, 
                  config.train.resume, config.train.restore_optim,
                  config.train.snapshot, config.train.backbone_path)


