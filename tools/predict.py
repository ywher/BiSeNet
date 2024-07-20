import sys
sys.path.insert(0, '.')
import os
import argparse

os.chdir(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from tqdm import tqdm
import numpy as np
import cv2

import torch

from lib.models import model_factory
from configs import set_cfg_from_file
import lib.data.transform_cv2 as T 
from lib.logger import setup_logger
from utils.color_map import color_map

class Predictor(object):
    def __init__(self, config, weight_path):
        self.cfg = set_cfg_from_file(config)
        self.net = self.load_model(weight_path)
        self.to_tensor = T.ToTensor_Img(
            mean=self.cfg.rgb_mean, # Bev, rgb
            std=self.cfg.rgb_std,
        )
        
    def load_model(self, weight_path):
        net = model_factory[self.cfg.model_type](self.cfg.n_cats)
        net.aux_mode = 'pred'
        net.load_state_dict(torch.load(weight_path, map_location='cpu'))
        net.cuda()
        net.eval()
        print('load model from:', weight_path)
        print('net aux mode', net.aux_mode)
        return net
        
    def create_folder(self, out_folder):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
            
    def colorize_prediction(self, prediction_image):
        """
        给语义分割模型的预测结果图上色。

        参数：
        - prediction_image: 预测结果图像的 NumPy 数组，其元素为类别标签。
        - color_map: 字典, 包含类别标签与对应颜色的映射关系。

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
    def predict(self, img):
        img = self.to_tensor(img)
        img = img.cuda().unsqueeze(0)
        with torch.no_grad():
            out = self.net(img).squeeze(0)
        out = out.cpu().numpy().astype(np.uint8)
        return out

    def predict_folder(self, dataset, img_folder, out_folder):
        self.color_map = color_map[dataset]
        
        self.create_folder(out_folder)
        trainid_folder = os.path.join(out_folder, 'trainid')
        self.create_folder(trainid_folder)
        color_folder = os.path.join(out_folder, 'color')
        self.create_folder(color_folder)

        print('Predicting total images in the folder:', len(os.listdir(img_folder)))
        for img_name in tqdm(os.listdir(img_folder)):
            img_path = os.path.join(img_folder, img_name)
            img = cv2.imread(img_path)[:, :, ::-1].copy()
            out = self.predict(img)
            cv2.imwrite(os.path.join(trainid_folder, img_name), out)
            color_out = self.colorize_prediction(out)
            cv2.imwrite(os.path.join(color_folder, img_name), color_out)

    def predict_image(self, img_path, out_folder):
        self.create_folder(out_folder)
        trainid_folder = os.path.join(out_folder, 'trainid')
        self.create_folder(trainid_folder)
        color_folder = os.path.join(out_folder, 'color')
        self.create_folder(color_folder)
        
        img = cv2.imread(img_path)
        out = self.predict(img)
        cv2.imwrite(os.path.join(trainid_folder, 'output.png'), out)
        color_out = self.colorize_prediction(out)
        cv2.imwrite(os.path.join(color_folder, 'color_output.png'), color_out)

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--config', dest='config', type=str, default='configs/bisenetv1_bev20234_1024.py')
    args.add_argument('--weight_path', dest='weight_path', type=str, default='res/bev_20234_1024_4witers/model_final.pth')
    args.add_argument('--image_folder', dest='image_folder', type=str, default='datasets/bev_20234_1024/image/val', help='the path to the image to be predicted')
    args.add_argument('--output_folder', dest='output_folder', type=str, default='res/bev_20234_1024_4witers/pred', help='the path to save the output')
    args.add_argument('--dataset', type=str, default='bev_20234')
    return args.parse_args()

def main():
    args = get_args()
    cfg = set_cfg_from_file(args.config)
    # setup_logger(cfg.respth)

    predictor = Predictor(args.config, args.weight_path)
    predictor.predict_folder(args.dataset, args.image_folder, args.output_folder)

if __name__ == '__main__':
    main()
