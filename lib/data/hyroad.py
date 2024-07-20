#!/usr/bin/python
# -*- encoding: utf-8 -*-


import lib.data.transform_cv2 as T
from lib.data.base_dataset import BaseDataset


class HYRoad(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train', norm={'mean':(0.4753, 0.4882, 0.4546), 'std':(0.1994, 0.2052, 0.2469)}, return_img_name=False):
        super(HYRoad, self).__init__(
                dataroot, annpath, trans_func, mode, norm, return_img_name)
        self.lb_ignore = 255
        self.norm_cfg = norm

        self.to_tensor = T.ToTensor(
            mean=self.norm_cfg['mean'], # Bev, rgb
            std=self.norm_cfg['std'],
        )


