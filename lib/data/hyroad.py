#!/usr/bin/python
# -*- encoding: utf-8 -*-


import lib.data.transform_cv2 as T
from lib.data.base_dataset import BaseDataset


class HYRoad(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(HYRoad, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.lb_ignore = 255

        self.to_tensor = T.ToTensor(
            mean=(0.4753, 0.4882, 0.4546), # Bev, rgb
            std=(0.1994, 0.2052, 0.2469),
        )


