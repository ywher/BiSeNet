#!/usr/bin/python
# -*- encoding: utf-8 -*-


import lib.data.transform_cv2 as T
from lib.data.base_dataset import BaseDataset


class Bev(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train', norm={'mean':(0.3755, 0.3719, 0.3771), 'std':(0.1465, 0.1529, 0.1504)}, return_img_name=False):
        super(Bev, self).__init__(
                dataroot, annpath, trans_func, mode, norm, return_img_name)
        self.lb_ignore = 255
        self.norm_cfg = norm

        self.to_tensor = T.ToTensor(
            mean=self.norm_cfg['mean'], # Bev, rgb
            std=self.norm_cfg['std'],
        )


