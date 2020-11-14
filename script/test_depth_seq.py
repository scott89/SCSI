from config.config_test_depth_seq import config
from network.packnet3d.DepthResNet import DepthResNet
import torch
import torch.nn as nn
from core.datasets import *
from core.datasets.transforms import Transform
from torch.utils.data import DataLoader, DistributedSampler
from utils.misc import sample_to_cuda, resize
from core.validator_seq import predict_depth, evaluate_depth, reduce_metrics, print_results, compute_scale

def test_depth(config):
    # network
    disp_net = DepthResNet('18').to(config.gpu)
    ckpt = torch.load(config.snapshot, map_location='cpu')
    disp_net.load_state_dict(ckpt['disp_net'])

    # dataset
    data_transform = Transform(config.dataset.transform)
    data_loaders = []
    for data_file in config.dataset.data_file:

        dataset = eval(config.dataset.name)(data_path = config.dataset.data_path,
                                            data_file = data_file,
                                            data_transform = data_transform,
                                            with_context=False,
                                            with_depth = True,
                                            with_pose = False
                                           )
        data_loader = DataLoader(dataset, batch_size=1, num_workers=config.dataset.num_workers,
                                 pin_memory=True, shuffle=False)
        data_loaders.append(data_loader)
    disp_net.eval()
    metrics = []
    for data_loader in data_loaders:
        depths = []
        gts = []
        for batch in data_loader:
            batch = sample_to_cuda(batch, config.gpu)
            with torch.no_grad():
                depth, gt = predict_depth(disp_net, batch)
            depths.append(depth)
            gts.append(gt)
        depth = torch.cat(depths, 0)
        gt = torch.cat(gts, 0)
        scale = compute_scale(gt, depth)
        for gt, depth in zip(gts, depths):
            metrics.append(evaluate_depth(gt, depth, scale))

    metrics = reduce_metrics(metrics)
    print_results(metrics)




if __name__ == '__main__':
    test_depth(config)
