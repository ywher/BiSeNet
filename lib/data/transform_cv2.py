#!/usr/bin/python
# -*- encoding: utf-8 -*-


import random
import math

import numpy as np
import cv2
import torch

### for debug
# import os
# import sys
# sys.path.insert(0, '.')
# sys.path.append('/media/ywh/pool1/yanweihao/projects/segmentation/real-time/BiSeNet')
# from utils.color_map import color_map as color_maps
# from utils.color_prediction import colorize_prediction_cv2


class RandomResizedCrop(object):
    '''
    size should be a tuple of (H, W)
    '''
    def __init__(self, scales=(0.5, 1.), size=(384, 384)):
        self.scales = scales
        self.size = size

    def __call__(self, im_lb):
        if self.size is None:
            return im_lb

        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]

        crop_h, crop_w = self.size
        scale = np.random.uniform(min(self.scales), max(self.scales))
        im_h, im_w = [math.ceil(el * scale) for el in im.shape[:2]]
        im = cv2.resize(im, (im_w, im_h))
        lb = cv2.resize(lb, (im_w, im_h), interpolation=cv2.INTER_NEAREST)

        if (im_h, im_w) == (crop_h, crop_w): return dict(im=im, lb=lb)
        pad_h, pad_w = 0, 0
        if im_h < crop_h:
            pad_h = (crop_h - im_h) // 2 + 1
        if im_w < crop_w:
            pad_w = (crop_w - im_w) // 2 + 1
        if pad_h > 0 or pad_w > 0:
            im = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
            lb = np.pad(lb, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=255)

        im_h, im_w, _ = im.shape
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))
        return dict(
            im=im[sh:sh+crop_h, sw:sw+crop_w, :].copy(),
            lb=lb[sh:sh+crop_h, sw:sw+crop_w].copy()
        )


class RandomRotate(object):
    '''
    Randomly rotate an image and its corresponding semantic label.
    The image will be filled with (0,0,0) in empty areas, and the label will be filled with 255.

    Parameters:
    angle (int or float): Maximum rotation angle in degrees. Rotation will be randomly chosen
                          between -angle and +angle.
    '''
    def __init__(self, angle=180, im_fill=(0, 0, 0), lb_fill=255):
        self.angle = angle
        self.im_fill = im_fill
        self.lb_fill = lb_fill
        self.p = 0.5
        
    def __call__(self, im_lb):
        if np.random.random() < self.p:
            return im_lb
        
        im, lb = im_lb['im'], im_lb['lb']
        
        # Ensure the image and label have the same dimensions
        assert im.shape[:2] == lb.shape[:2]
        
        # Generate a random rotation angle between -self.angle and +self.angle
        angle = np.random.uniform(-self.angle, self.angle)
        
        # Get image dimensions
        h, w = im.shape[:2]
        
        # Calculate the rotation matrix
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        
        # Apply the affine transformation (rotation) with filling values
        im = cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=self.im_fill)
        lb = cv2.warpAffine(lb, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.lb_fill)
        
        # Return the rotated image and label in the original dictionary format
        return dict(im=im, lb=lb)



class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im_lb):
        if np.random.random() < self.p:
            return im_lb
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]
        return dict(
            im=im[:, ::-1, :],
            lb=lb[:, ::-1],
        )



class ColorJitter(object):

    def __init__(self, brightness=None, contrast=None, saturation=None):
        if not brightness is None and brightness >= 0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast >= 0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation >= 0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]
        if not self.brightness is None:
            rate = np.random.uniform(*self.brightness)
            im = self.adj_brightness(im, rate)
        if not self.contrast is None:
            rate = np.random.uniform(*self.contrast)
            im = self.adj_contrast(im, rate)
        if not self.saturation is None:
            rate = np.random.uniform(*self.saturation)
            im = self.adj_saturation(im, rate)
        return dict(im=im, lb=lb,)

    def adj_saturation(self, im, rate):
        M = np.float32([
            [1+2*rate, 1-rate, 1-rate],
            [1-rate, 1+2*rate, 1-rate],
            [1-rate, 1-rate, 1+2*rate]
        ])
        shape = im.shape
        im = np.matmul(im.reshape(-1, 3), M).reshape(shape)/3
        im = np.clip(im, 0, 255).astype(np.uint8)
        return im

    def adj_brightness(self, im, rate):
        table = np.array([
            i * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[im]

    def adj_contrast(self, im, rate):
        table = np.array([
            74 + (i - 74) * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[im]


class ToTensor(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        im = im.sub_(mean).div_(std).clone()
        if not lb is None:
            lb = torch.from_numpy(lb.astype(np.int64).copy()).clone()
        return dict(im=im, lb=lb)
    
class ToTensor_Img(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im):
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        im = im.sub_(mean).div_(std).clone()

        return im


class Compose(object):

    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb


class TransformationTrain(object):

    def __init__(self, scales, cropsize, rotate=0):
        if rotate > 0:
            self.trans_func = Compose([
                RandomResizedCrop(scales, cropsize),
                RandomRotate(rotate),
                RandomHorizontalFlip(),
                ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4
                ),
            ])
        else:
            self.trans_func = Compose([
                RandomResizedCrop(scales, cropsize),
                RandomHorizontalFlip(),
                ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4
                ),
            ])

    def __call__(self, im_lb):
        im_lb = self.trans_func(im_lb)
        return im_lb


class TransformationVal(object):

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        return dict(im=im, lb=lb)



if __name__ == '__main__':
    img_path = "images/img.png"
    lb_path = "images/lb.png"
    img = cv2.imread(img_path)
    lb = cv2.imread(lb_path, cv2.IMREAD_GRAYSCALE)
    im_lb_dict = {'im':img, 'lb':lb}
    color_mp = color_maps['bev_20234']
    
    random_rotate = RandomRotate()
    rotate_img_lb = random_rotate(im_lb_dict)
    rotate_img = rotate_img_lb['im']
    rotate_lb = rotate_img_lb['lb']
    color_rotate_lb = colorize_prediction_cv2(rotate_lb, color_mp)[:,:,::-1]
    cv2.imwrite("images/rotate_img.png", rotate_img)
    cv2.imwrite("images/rotate_lb.png", rotate_lb)
    cv2.imwrite("images/color_rotate_lb.png", color_rotate_lb)

