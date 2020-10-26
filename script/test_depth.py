from config.config_test_depth import config
from network.packnet3d.DepthResNet import DepthResNet
import torch
import torch.nn as nn
from core.datasets import *
from core.datasets.transforms import Transform
from torch.utils.data import DataLoader, DistributedSampler
from utils.misc import sample_to_cuda, resize
from core.validator import evaluate_depth, reduce_metrics, print_results

def test_depth(config):
    # network
    disp_net = DepthResNet('18').to(config.gpu)
    ckpt = torch.load(config.snapshot, map_location='cpu')
    disp_net.load_state_dict(ckpt['disp_net'])

    # dataset
    data_transform = Transform(config.dataset.transform)
    dataset = eval(config.dataset.name)(data_path = config.dataset.data_path,
                                        data_file = config.dataset.data_file,
                                        data_transform = data_transform,
                                        with_context=False,
                                        with_depth = True,
                                        with_pose = False
                                       )
    data_loader = DataLoader(dataset, batch_size=1, num_workers=config.dataset.num_workers,
                             pin_memory=True, shuffle=False)
    outputs = []
    disp_net.eval()
    for batch in data_loader:
        batch = sample_to_cuda(batch, config.gpu)
        with torch.no_grad():
            output = evaluate_depth(disp_net, batch)
        outputs.append(output)
    metrics = reduce_metrics(outputs)
    print_results(metrics)




if __name__ == '__main__':
    test_depth(config)
