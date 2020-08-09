import tensorboardX
import os
from os.path import join


def build_summary_writer(config):
    train_log_path = join(config.train.output_path, 'train_summary')
    val_log_path = join(config.train.output_path, 'val_summary')
    if not os.path.isdir(train_log_path):
        os.makedirs(train_log_path)
    if not os.path.isdir(val_log_path):
        os.makedirs(val_log_path)

    train_summary_writer = tensorboardX.SummaryWriter(logdir=train_log_path)
    val_summary_writer = tensorboardX.SummaryWriter(logdir=val_log_path)
    return train_summary_writer, val_summary_writer

