import torch
import torch.nn.functional as F
from utils.misc import sample_to_cuda, disp2depth, write_val_summary_helper
from collections import OrderedDict

EVAL_MODES = ['depth', 'depth_pp', 'detph_gt', 'depth_pp_gt']
METRICS = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']

def post_process_inv_depth(disp, disp_flipped):
    B, C, H, W = disp.shape
    disp_hat = torch.flip(disp_flipped, [3])
    disp_fused = (disp + disp_hat) / 2
    xs = torch.linspace(0., 1., W, device=disp.device,
                        dtype=disp.dtype).repeat(B, C, H, 1)
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    mask_hat = torch.flip(mask, [3])
    return mask_hat * disp + mask * disp_hat + \
           (1.0 - mask - mask_hat) * disp_fused


def compute_depth_metrics(gt, pred, use_gt_scale=True, min_depth=0, max_depth=80):
    """
    Compute depth metrics from predicted and ground-truth depth maps

    Parameters
    ----------
    gt : torch.Tensor [B,1,H,W]
        Ground-truth depth map
    pred : torch.Tensor [B,1,H,W]
        Predicted depth map
    use_gt_scale : bool
        True if ground-truth median-scaling is to be used

    Returns
    -------
    metrics : torch.Tensor [7]
        Depth metrics (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)
    """
    crop = True

    # Initialize variables
    batch_size, _, gt_height, gt_width = gt.shape
    abs_diff = abs_rel = sq_rel = rmse = rmse_log = a1 = a2 = a3 = 0.0
    # Interpolate predicted depth to ground-truth resolution
    #pred = interpolate_image(pred, gt.shape, mode='bilinear', align_corners=True)
    pred = F.interpolate(pred, gt.shape[2:], mode='bilinear', align_corners=True)
    # If using crop
    if crop:
        crop_mask = torch.zeros(gt.shape[-2:]).byte().type_as(gt)
        y1, y2 = int(0.40810811 * gt_height), int(0.99189189 * gt_height)
        x1, x2 = int(0.03594771 * gt_width), int(0.96405229 * gt_width)
        crop_mask[y1:y2, x1:x2] = 1
    # For each depth map
    for pred_i, gt_i in zip(pred, gt):
        gt_i, pred_i = torch.squeeze(gt_i), torch.squeeze(pred_i)
        # Keep valid pixels (min/max depth and crop)
        valid = (gt_i > min_depth) & (gt_i < max_depth)
        valid = valid & crop_mask.bool() if crop else valid
        # Stop if there are no remaining valid pixels
        if valid.sum() == 0:
            continue
        # Keep only valid pixels
        gt_i, pred_i = gt_i[valid], pred_i[valid]
        # Ground-truth median scaling if needed
        if use_gt_scale:
            pred_i = pred_i * torch.median(gt_i) / torch.median(pred_i)
        # Clamp predicted depth values to min/max values
        pred_i = pred_i.clamp(min_depth, max_depth)

        # Calculate depth metrics

        thresh = torch.max((gt_i / pred_i), (pred_i / gt_i))
        a1 += (thresh < 1.25     ).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        diff_i = gt_i - pred_i
        abs_diff += torch.mean(torch.abs(diff_i))
        abs_rel += torch.mean(torch.abs(diff_i) / gt_i)
        sq_rel += torch.mean(diff_i ** 2 / gt_i)
        rmse += torch.sqrt(torch.mean(diff_i ** 2))
        rmse_log += torch.sqrt(torch.mean((torch.log(gt_i) -
                                           torch.log(pred_i)) ** 2))
    # Return average values for each metric
    return torch.tensor([metric / batch_size for metric in
        [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]]).type_as(gt)

def evaluate_depth(disp_net, batch):
        """Evaluate batch to produce depth metrics."""
        # Get predicted depth
        disp = disp_net(batch['rgb'])[0][0]
        depth = disp2depth(disp)
        # Post-process predicted depth
        batch['rgb'] = torch.flip(batch['rgb'], [3])
        disp_flipped = disp_net(batch['rgb'])[0][0]
        disp_pp = post_process_inv_depth(
            disp, disp_flipped)
        depth_pp = disp2depth(disp_pp)
        batch['rgb'] = torch.flip(batch['rgb'], [3])
        # Calculate predicted metrics
        metrics = OrderedDict()
        for mode in EVAL_MODES:
            metrics[mode] = compute_depth_metrics(
                gt=batch['depth'],
                pred=depth_pp if 'pp' in mode else depth,
                use_gt_scale='gt' in mode)
        # Return metrics and extra information
        return {
            'metrics': metrics,
            'disp': disp_pp,
        }


def reduce_metrics(outputs):
    metrics = dict()
    for output in outputs:
        for k in output['metrics']:
            if k not in metrics.keys():
                metrics[k] = output['metrics'][k].clone()
            else:
                metrics[k] += output['metrics'][k]
    for k in metrics:
        metrics[k] /= len(outputs)
    return metrics

def print_results(metrics):
    for mode in metrics:
        m = metrics[mode].cpu().numpy()
        s = '%s mode: '%mode
        for i, k in enumerate(METRICS):
            s += '%s: %f, '%(k, m[i])
        s = s[:-2]
        print(s)

def depth_validator(disp_net, val_dataloader, val_summary, epoch, global_step, gpu_id):
    disp_net.eval()
    outputs = []
    print('Val epoch #%d'%epoch)
    for batch in val_dataloader:
        batch = sample_to_cuda(batch, gpu_id)
        with torch.no_grad():
            output = evaluate_depth(disp_net, batch)
        outputs.append(output)
    metrics = reduce_metrics(outputs)
    print_results(metrics)
    write_val_summary_helper(val_summary, batch['rgb_original'], 
                             disp2depth(batch['depth']), 
                             output['disp'],
                             metrics['depth_pp_gt'],
                             METRICS,
                             global_step
                            )








