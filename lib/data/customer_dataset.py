#!/usr/bin/python
# -*- encoding: utf-8 -*-


import lib.data.transform_cv2 as T
from lib.data.base_dataset import BaseDataset


class CustomerDataset(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train', norm={'mean':(0.3257, 0.3690, 0.3223), 'std':(0.2112, 0.2148, 0.2115)}, return_img_name=False):
        super(CustomerDataset, self).__init__(
                dataroot, annpath, trans_func, mode, norm, return_img_name)
        self.lb_ignore = 255
        self.norm_cfg = norm

        self.to_tensor = T.ToTensor(
            mean=self.norm_cfg['mean'], # rgb
            std=self.norm_cfg['std'],
        )


