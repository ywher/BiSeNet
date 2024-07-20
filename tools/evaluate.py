#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import logging
import argparse
import math
from tabulate import tabulate

os.chdir(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from tqdm import tqdm
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.logger import setup_logger
from lib.data import get_data_loader
from utils.color_map import color_map
from utils.classes import CLASSES


def get_round_size(size, divisor=32):
    return [math.ceil(el / divisor) * divisor for el in size]


class SizePreprocessor(object):

    def __init__(self, shape=None, shortside=None, longside=None):
        self.shape = shape
        self.shortside = shortside
        self.longside = longside

    def __call__(self, imgs):
        new_size = None
        if not self.shape is None:
            new_size = self.shape
        elif not self.shortside is None:
            h, w = imgs.size()[2:]
            ss = self.shortside
            if h < w: h, w = ss, int(ss / h * w)
            else: h, w = int(ss / w * h), ss
            new_size = h, w
        elif not self.longside is None: # long size limit
            h, w = imgs.size()[2:]
            if max(h, w) > self.longside:
                ls = self.longside
                if h < w: h, w = int(ls / w * h), ls
                else: h, w = ls, int(ls / h * w)
                new_size = h, w

        if not new_size is None:
            imgs = F.interpolate(imgs, size=new_size,
                    mode='bilinear', align_corners=False)
        return imgs



class Metrics(object):

    def __init__(self, n_classes, lb_ignore=255):
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.confusion = torch.zeros((n_classes, n_classes)).cuda().detach()

    @torch.no_grad()
    def update(self, preds, label):
        keep = label != self.lb_ignore
        preds, label = preds[keep], label[keep]
        self.confusion += torch.bincount(
                label * self.n_classes + preds,
                minlength=self.n_classes ** 2
                ).view(self.n_classes, self.n_classes)

    @torch.no_grad()
    def compute_metrics(self,):
        if dist.is_initialized():
            dist.all_reduce(self.confusion, dist.ReduceOp.SUM)

        confusion = self.confusion
        weights = confusion.sum(dim=1) / confusion.sum()
        tps = confusion.diag()
        fps = confusion.sum(dim=0) - tps
        fns = confusion.sum(dim=1) - tps

        # iou and fw miou
        #  ious = confusion.diag() / (confusion.sum(dim=0) + confusion.sum(dim=1) - confusion.diag() + 1)
        ious = tps / (tps + fps + fns + 1)
        miou = ious.nanmean()
        fw_miou = torch.sum(weights * ious)

        eps = 1e-6
        # macro f1 score
        macro_precision = tps / (tps + fps + 1)
        macro_recall = tps / (tps + fns + 1)
        f1_scores = (2 * macro_precision * macro_recall) / (
                macro_precision + macro_recall + eps)
        macro_f1 = f1_scores.nanmean(dim=0)

        # micro f1 score
        tps_ = tps.sum(dim=0)
        fps_ = fps.sum(dim=0)
        fns_ = fns.sum(dim=0)
        micro_precision = tps_ / (tps_ + fps_ + 1)
        micro_recall = tps_ / (tps_ + fns_ + 1)
        micro_f1 = (2 * micro_precision * micro_recall) / (
                micro_precision + micro_recall + eps)

        metric_dict = dict(
                weights=weights.tolist(),
                ious=ious.tolist(),
                miou=miou.item(),
                fw_miou=fw_miou.item(),
                f1_scores=f1_scores.tolist(),
                macro_f1=macro_f1.item(),
                micro_f1=micro_f1.item(),
                )
        return metric_dict



class MscEvalV0(object):

    def __init__(self, n_classes, scales=(0.5, ), flip=False, lb_ignore=255, size_processor=None, save_pred=False, save_root=None, color_map={}):
        self.n_classes = n_classes
        self.scales = scales
        self.flip = flip
        self.ignore_label = lb_ignore
        self.sp = size_processor
        self.metric_observer = Metrics(n_classes, lb_ignore)
        self.save_pred = save_pred
        self.save_root = save_root
        if self.save_pred and self.save_root is not None:
            self.save_root = os.path.join(self.save_root, 'pred')
            self.create_folder(self.save_root)
            self.create_folder(os.path.join(self.save_root, 'trainid'))
            self.create_folder(os.path.join(self.save_root, 'color'))
            self.color_map = color_map
    
    def create_folder(self, out_folder):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
    
    def colorize_prediction(self, prediction_image):
        """
        color the prediction of semantic segmentation model

        参数：
        - prediction_image: prediction in NumPy array

        返回值：
        - colored_image: 上色后的 numpy 对象, in BGR format.
        """
        # 将PIL Image转换为NumPy数组
        prediction_array = np.array(prediction_image)

        # 获取图像大小
        height, width = prediction_array.shape

        # 创建一个新的数组来存储上色后的图像
        colored_array = np.zeros((height, width, 3), dtype=np.uint8)

        # 根据颜色映射给预测结果图上色
        for label, color in self.color_map.items():
            # 找到与类别标签对应的像素点，并将其设为相应的颜色
            colored_array[prediction_array == label] = color

        colored_array = colored_array[:, :, ::-1]
        
        return colored_array

    @torch.no_grad()
    def __call__(self, net, dl):
        ## evaluate
        n_classes = self.n_classes
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, data in diter:
            if self.save_pred:
                (imgs, label, names) = data
            else:
                (imgs, label) = data
            imgs = self.sp(imgs)
            N, _, H, W = imgs.shape
            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            probs = torch.zeros(
                    (N, n_classes, *size),
                    dtype=torch.float32).cuda().detach()
            for scale in self.scales:
                sH, sW = int(scale * H), int(scale * W)
                sH, sW = get_round_size((sH, sW))
                im_sc = F.interpolate(imgs, size=(sH, sW),
                        mode='bilinear', align_corners=True)

                im_sc = im_sc.cuda()
                logits = net(im_sc)[0]
                logits = F.interpolate(logits, size=size,
                        mode='bilinear', align_corners=True)
                probs += torch.softmax(logits, dim=1)
                if self.flip:
                    im_sc = torch.flip(im_sc, dims=(3, ))
                    logits = net(im_sc)[0]
                    logits = torch.flip(logits, dims=(3, ))
                    logits = F.interpolate(logits, size=size,
                            mode='bilinear', align_corners=True)
                    probs += torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            self.metric_observer.update(preds, label)
            
            # save the prediction trainid
            if self.save_pred and self.save_root is not None:
                self.save_pred_trainid(preds, names)

        metric_dict = self.metric_observer.compute_metrics()
        return metric_dict
    
    def save_pred_trainid(self, preds, img_names):
        for pred, img_name in zip(preds, img_names):
            pred = pred.cpu().numpy().astype(np.uint8)
            pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
            save_path = osp.join(self.save_root, 'trainid', img_name)
            cv2.imwrite(save_path, pred)
            color_pred = self.colorize_prediction(pred[:, :, 0])
            save_path = osp.join(self.save_root, 'color', img_name)
            cv2.imwrite(save_path, color_pred)


class MscEvalCrop(object):

    def __init__(
        self,
        n_classes,
        cropsize=1024,
        cropstride=2./3,
        flip=True,
        scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
        lb_ignore=255,
        size_processor=None,
        save_pred=False,
    ):
        self.n_classes = n_classes
        self.scales = scales
        self.ignore_label = lb_ignore
        self.flip = flip
        self.sp = size_processor
        self.save_pred = save_pred
        self.metric_observer = Metrics(n_classes, lb_ignore)

        self.cropsize = cropsize if isinstance(cropsize, (list, tuple)) else (cropsize, cropsize)
        self.cropstride = cropstride


    def pad_tensor(self, inten):
        N, C, H, W = inten.size()
        cropH, cropW = self.cropsize
        if cropH < H and cropW < W: return inten, [0, H, 0, W]
        padH, padW = max(cropH, H), max(cropW, W)
        outten = torch.zeros(N, C, padH, padW).cuda()
        outten.requires_grad_(False)
        marginH, marginW = padH - H, padW - W
        hst, hed = marginH // 2, marginH // 2 + H
        wst, wed = marginW // 2, marginW // 2 + W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]


    def eval_chip(self, net, crop):
        prob = net(crop)[0].softmax(dim=1)
        if self.flip:
            crop = torch.flip(crop, dims=(3,))
            prob += net(crop)[0].flip(dims=(3,)).softmax(dim=1)
            prob = torch.exp(prob)
        return prob


    def crop_eval(self, net, im, n_classes):
        cropH, cropW = self.cropsize
        stride_rate = self.cropstride
        im, indices = self.pad_tensor(im)
        N, C, H, W = im.size()

        strdH = math.ceil(cropH * stride_rate)
        strdW = math.ceil(cropW * stride_rate)
        n_h = math.ceil((H - cropH) / strdH) + 1
        n_w = math.ceil((W - cropW) / strdW) + 1
        prob = torch.zeros(N, n_classes, H, W).cuda()
        prob.requires_grad_(False)
        for i in range(n_h):
            for j in range(n_w):
                stH, stW = strdH * i, strdW * j
                endH, endW = min(H, stH + cropH), min(W, stW + cropW)
                stH, stW = endH - cropH, endW - cropW
                chip = im[:, :, stH:endH, stW:endW]
                prob[:, :, stH:endH, stW:endW] += self.eval_chip(net, chip)
        hst, hed, wst, wed = indices
        prob = prob[:, :, hst:hed, wst:wed]
        return prob


    def scale_crop_eval(self, net, im, scale, size, n_classes):
        N, C, H, W = im.size()
        new_hw = [int(H * scale), int(W * scale)]
        im = F.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(net, im, n_classes)
        prob = F.interpolate(prob, size, mode='bilinear', align_corners=True)
        return prob


    @torch.no_grad()
    def __call__(self, net, dl):

        n_classes = self.n_classes
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        hist.requires_grad_(False)
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))

        for i, data in diter:
            if self.save_pred:
                (imgs, label, _) = data
            else:
                (imgs, label) = data
            imgs = imgs.cuda()
            imgs = self.sp(imgs)
            label = label.squeeze(1).cuda()
            N, *size = label.size()

            probs = torch.zeros((N, n_classes, *size)).cuda()
            probs.requires_grad_(False)
            for sc in self.scales:
                probs += self.scale_crop_eval(net, imgs, sc, size, n_classes)
            torch.cuda.empty_cache()
            preds = torch.argmax(probs, dim=1)
            self.metric_observer.update(preds, label)

        metric_dict = self.metric_observer.compute_metrics()
        return metric_dict


def print_res_table(tname, heads, weights, metric, cat_metric, class_names=None):
    heads = [tname, 'ratio'] + heads
    lines = []
    for k, v in metric.items():
        line = [k, '-'] + [f'{el:.6f}' for el in v]
        lines.append(line)
    cat_res = [weights,] + cat_metric
    if class_names is not None:
        cat_res = [
                [f'{class_names[idx]}',] + [f'{el:.6f}' for el in group]
                for idx,group in enumerate(zip(*cat_res))]
    else:
        cat_res = [
                [f'cat {idx}',] + [f'{el:.6f}' for el in group]
                for idx,group in enumerate(zip(*cat_res))]
    content = cat_res + lines
    return heads, content


@torch.no_grad()
def eval_model(cfg, net, save_pred):
    org_aux = net.aux_mode
    net.aux_mode = 'eval'
    net.eval()

    is_dist = dist.is_initialized()
    dl = get_data_loader(cfg, mode='val',save_pred=save_pred)
    lb_ignore = dl.dataset.lb_ignore

    heads, mious, fw_mious, cat_ious = [], [], [], []
    f1_scores, macro_f1, micro_f1 = [], [], []
    logger = logging.getLogger()

    size_processor = SizePreprocessor(
            cfg.get('eval_start_shape'),
            cfg.get('eval_start_shortside'),
            cfg.get('eval_start_longside'),
            )  # None None None

    color_dict = color_map[cfg.color_dataset]
    single_scale = MscEvalV0(
            n_classes=cfg.n_cats,
            scales=(1., ),
            flip=False,
            lb_ignore=lb_ignore,
            size_processor=size_processor,
            save_pred=save_pred,
            save_root=cfg.respth,
            color_map=color_dict
    )
    logger.info('compute single scale metrics')
    metrics = single_scale(net, dl)
    heads.append('ss')
    mious.append(metrics['miou'])
    fw_mious.append(metrics['fw_miou'])
    cat_ious.append(metrics['ious'])
    f1_scores.append(metrics['f1_scores'])
    macro_f1.append(metrics['macro_f1'])
    micro_f1.append(metrics['micro_f1'])

    single_crop = MscEvalCrop(
            n_classes=cfg.n_cats,
            cropsize=cfg.eval_crop,
            cropstride=2. / 3,
            flip=False,
            scales=(1., ),
            lb_ignore=lb_ignore,
            size_processor=size_processor,
            save_pred=save_pred,
    )
    logger.info('compute single scale crop metrics')
    metrics = single_crop(net, dl)
    heads.append('ssc')
    mious.append(metrics['miou'])
    fw_mious.append(metrics['fw_miou'])
    cat_ious.append(metrics['ious'])
    f1_scores.append(metrics['f1_scores'])
    macro_f1.append(metrics['macro_f1'])
    micro_f1.append(metrics['micro_f1'])

    ms_flip = MscEvalV0(
            n_classes=cfg.n_cats,
            scales=cfg.eval_scales,
            flip=True,
            lb_ignore=lb_ignore,
            size_processor=size_processor,
            save_pred=save_pred,
    )
    logger.info('compute multi scale flip metrics')
    metrics = ms_flip(net, dl)
    heads.append('msf')
    mious.append(metrics['miou'])
    fw_mious.append(metrics['fw_miou'])
    cat_ious.append(metrics['ious'])
    f1_scores.append(metrics['f1_scores'])
    macro_f1.append(metrics['macro_f1'])
    micro_f1.append(metrics['micro_f1'])

    ms_flip_crop = MscEvalCrop(
            n_classes=cfg.n_cats,
            cropsize=cfg.eval_crop,
            cropstride=2. / 3,
            flip=True,
            scales=cfg.eval_scales,
            lb_ignore=lb_ignore,
            size_processor=size_processor,
            save_pred=save_pred,
    )
    logger.info('compute multi scale flip crop metrics')
    metrics = ms_flip_crop(net, dl)
    heads.append('msfc')
    mious.append(metrics['miou'])
    fw_mious.append(metrics['fw_miou'])
    cat_ious.append(metrics['ious'])
    f1_scores.append(metrics['f1_scores'])
    macro_f1.append(metrics['macro_f1'])
    micro_f1.append(metrics['micro_f1'])

    weights = metrics['weights']
    
    class_names = CLASSES.get(cfg.color_dataset, None)
    metric = dict(mious=mious, fw_mious=fw_mious)
    iou_heads, iou_content = print_res_table('iou', heads,
            weights, metric, cat_ious, class_names=class_names)
    metric = dict(macro_f1=macro_f1, micro_f1=micro_f1)
    f1_heads, f1_content = print_res_table('f1 score', heads,
            weights, metric, f1_scores, class_names=class_names)

    net.aux_mode = org_aux
    return iou_heads, iou_content, f1_heads, f1_content


def evaluate(cfg, weight_pth, save_pred=False):
    logger = logging.getLogger()

    ## model
    logger.info('setup and restore model')
    net = model_factory[cfg.model_type](cfg.n_cats)
    net.load_state_dict(torch.load(weight_pth, map_location='cpu'))
    net.cuda()

    #  if dist.is_initialized():
    #      local_rank = dist.get_rank()
    #      net = nn.parallel.DistributedDataParallel(
    #          net,
    #          device_ids=[local_rank, ],
    #          output_device=local_rank
    #      )

    ## evaluator
    iou_heads, iou_content, f1_heads, f1_content = eval_model(cfg, net, save_pred)
    logger.info('\neval results of f1 score metric:')
    logger.info('\n' + tabulate(f1_content, headers=f1_heads, tablefmt='orgtbl'))
    logger.info('\neval results of miou metric:')
    logger.info('\n' + tabulate(iou_content, headers=iou_heads, tablefmt='orgtbl'))


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv1_bev2024.py',)
    parse.add_argument('--weight-path', dest='weight_pth', type=str, default='res/bev_2024/model_final.pth',)
    parse.add_argument('--save_pred', action='store_true', default=False)
    return parse.parse_args()


def main():
    args = parse_args()
    cfg = set_cfg_from_file(args.config)
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger(f'{cfg.model_type}-{cfg.dataset.lower()}-eval', cfg.respth)
    evaluate(cfg, args.weight_pth, args.save_pred)


if __name__ == "__main__":
    main()
