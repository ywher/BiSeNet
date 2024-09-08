#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import json
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from tensorboardX import SummaryWriter

from lib.models import model_factory
from configs import set_cfg_from_file, cvt_cfg_dict_to_json
from lib.data import get_data_loader
from evaluate import eval_model, eval_model_single_scale
from lib.losses import OhemCELoss, CE_DiceLoss, LovaszSoftmax
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg
# torch.autograd.set_detect_anomaly(True)


## fix all random seeds
#  torch.manual_seed(123)
#  torch.cuda.manual_seed(123)
#  np.random.seed(123)
#  random.seed(123)
#  torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str,
            default='configs/bisenetv2.py',)
    # parse.add_argument('--finetune-from', type=str, default=None,)
    parse.add_argument('--finetune-from', action='store_true', default=False)
    parse.add_argument('--local_rank', type=int, default=0,)
    return parse.parse_args()

args = parse_args()
cfg = set_cfg_from_file(args.config)


def set_model(lb_ignore=255):
    logger = logging.getLogger()
    net = model_factory[cfg.model_type](cfg.n_cats)
    # if not args.finetune_from is None:
    if args.finetune_from:
        logger.info(f'load pretrained weights from {cfg.pretrained}')
        # msg = net.load_state_dict(torch.load(args.finetune_from,
        #     map_location='cpu'), strict=False)
        msg = net.load_pretrained_model(cfg.pretrained, cfg.rm_layer_names)
        logger.info('\tmissing keys: ' + json.dumps(msg.missing_keys))
        logger.info('\tunexpected keys: ' + json.dumps(msg.unexpected_keys))
    if cfg.use_sync_bn: net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net.train()
    
    return net

def set_loss(lb_ignore=255):
    if cfg.loss_type == 'OhemCELoss':
        criteria_pre = OhemCELoss(0.7, lb_ignore)
        criteria_aux = [OhemCELoss(0.7, lb_ignore)
                for _ in range(cfg.num_aux_heads)]
    elif cfg.loss_type == 'CE_DiceLoss':
        criteria_pre = CE_DiceLoss(ignore_index=lb_ignore)
        criteria_aux = [CE_DiceLoss(ignore_index=lb_ignore)
                for _ in range(cfg.num_aux_heads)]
    elif cfg.loss_type == 'LovaszSoftmax':
        criteria_pre = LovaszSoftmax(ignore_index=lb_ignore)
        criteria_aux = [LovaszSoftmax(ignore_index=lb_ignore)
                for _ in range(cfg.num_aux_heads)]
    else:
        raise NotImplementedError
    return criteria_pre, criteria_aux
    


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        #  wd_val = cfg.weight_decay
        wd_val = 0
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': wd_val},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': cfg.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=cfg.lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    return optim


def set_model_dist(net):
    local_rank = int(os.environ['LOCAL_RANK'])
    net = nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank, ],
        #  find_unused_parameters=True,
        output_device=local_rank
        )
    return net


def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters



def train():
    # save config into config respth
    cfg_dict = cvt_cfg_dict_to_json(cfg)
    with open(osp.join(cfg.respth, 'config.json'), 'w') as f:
        f.write(json.dumps(cfg_dict, indent=4))
        
    logger = logging.getLogger()
    
    if args.local_rank == 0:
        tensorboard_logger = SummaryWriter(osp.join(cfg.respth, 'tensorboard'))

    ## dataset
    dl = get_data_loader(cfg, mode='train')

    ## model
    net = set_model(dl.dataset.lb_ignore)
    
    ## loss
    criteria_pre, criteria_aux = set_loss(dl.dataset.lb_ignore)

    ## optimizer
    optim = set_optimizer(net)

    ## mixed precision training
    scaler = amp.GradScaler()

    ## ddp training
    net = set_model_dist(net)

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    ## train loop
    best_mIoU = 0
    for it, (im, lb) in enumerate(dl):
        im = im.cuda()
        lb = lb.cuda()
        # if args.local_rank == 0:
        #     logger.info(f'original im.shape: {im.shape}, lb.shape: {lb.shape}')
        if cfg.get('train_im_scale', False):
            im = F.interpolate(im, scale_factor=cfg.train_im_scale, mode='bilinear', align_corners=True)
        # if args.local_rank == 0:
        #     logger.info(f'new im.shape: {im.shape}, lb.shape: {lb.shape}')

        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        with amp.autocast(enabled=cfg.use_fp16):
            logits, *logits_aux = net(im)
            if logits.size()[-2:] != lb.size()[-2:]:
                logits = F.interpolate(logits, size=lb.size()[-2:], mode='bilinear', align_corners=True)
            for i in range(len(logits_aux)):
                if logits_aux[i].size()[-2:] != lb.size()[-2:]:
                    logits_aux[i] = F.interpolate(logits_aux[i], size=lb.size()[-2:], mode='bilinear', align_corners=True)
            loss_pre = criteria_pre(logits, lb)
            loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
            loss = loss_pre + sum(loss_aux)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        torch.cuda.synchronize()

        time_meter.update()
        loss_meter.update(loss.item())
        loss_pre_meter.update(loss_pre.item())
        _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]
        
        ## tensorboard logging
        if args.local_rank == 0:
            tensorboard_logger.add_scalar('loss', loss.item(), it + 1)
            tensorboard_logger.add_scalar('loss_pre', loss_pre.item(), it + 1)
            for i, lss in enumerate(loss_aux):
                tensorboard_logger.add_scalar(f'loss_aux{i}', lss.item(), it + 1)
            tensorboard_logger.add_scalar('lr', lr_schdr.get_lr()[0], it + 1)

        ## print training log message
        if (it + 1) % 100 == 0:
            lr = lr_schdr.get_lr()
            lr = sum(lr) / len(lr)
            print_log_msg(it, cfg.max_iter, lr, time_meter, loss_meter, loss_pre_meter, loss_aux_meters)
        
        if (it + 1) % cfg.eval_intervals == 0:
            logger.info('\nevaluating the final model')
            # torch.cuda.empty_cache()
            mIoU, iou_heads, iou_content, f1_heads, f1_content = eval_model_single_scale(cfg, net.module)
            net.module.train()
            logger.info('\neval results of f1 score metric:')
            logger.info('\n' + tabulate(f1_content, headers=f1_heads, tablefmt='orgtbl'))
            logger.info('\neval results of miou metric:')
            logger.info('\n' + tabulate(iou_content, headers=iou_heads, tablefmt='orgtbl'))
            if args.local_rank == 0:
                tensorboard_logger.add_scalar('mIoU', mIoU, it + 1)
            if mIoU > best_mIoU:
                best_mIoU = mIoU
                save_pth = osp.join(cfg.respth, 'model_best.pth')
                logger.info('\nsave models to {}'.format(save_pth))
                state = net.module.state_dict()
                if dist.get_rank() == 0: torch.save(state, save_pth)
            
        lr_schdr.step()

    ## dump the final model and evaluate the result
    save_pth = osp.join(cfg.respth, 'model_final.pth')
    logger.info('\nsave models to {}'.format(save_pth))
    state = net.module.state_dict()
    if dist.get_rank() == 0: torch.save(state, save_pth)

    logger.info('\nevaluating the final model')
    torch.cuda.empty_cache()
    iou_heads, iou_content, f1_heads, f1_content = eval_model(cfg, net.module)
    logger.info('\neval results of f1 score metric:')
    logger.info('\n' + tabulate(f1_content, headers=f1_heads, tablefmt='orgtbl'))
    logger.info('\neval results of miou metric:')
    logger.info('\n' + tabulate(iou_content, headers=iou_heads, tablefmt='orgtbl'))

    return


def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    if not osp.exists(cfg.respth): os.makedirs(cfg.respth, exist_ok=True)
    setup_logger(f'{cfg.model_type}-{cfg.dataset.lower()}-train', cfg.respth)
    train()


if __name__ == "__main__":
    main()
