import argparse
import os.path as osp
import sys
sys.path.insert(0, '.')

import torch

from lib.models import model_factory
from configs import set_cfg_from_file

torch.set_grad_enabled(False)

def get_args():
        parse = argparse.ArgumentParser()
        parse.add_argument('--config', dest='config', type=str,
                default='configs/bisenetv2.py',)
        parse.add_argument('--weight-path', dest='weight_pth', type=str,
                default='model_final.pth')
        parse.add_argument('--outpath', dest='out_pth', type=str,
                default='model.onnx')
        parse.add_argument('--aux-mode', dest='aux_mode', type=str,
                default='pred')
        args = parse.parse_args()
        return args

if __name__ == '__main__':
        args = get_args()
        cfg = set_cfg_from_file(args.config)
        if cfg.use_sync_bn: cfg.use_sync_bn = False

        net = model_factory[cfg.model_type](cfg.n_cats, aux_mode=args.aux_mode)
        net.load_state_dict(torch.load(args.weight_pth, map_location='cpu'), strict=False)
        net.eval()

        dummy_input = torch.randn(1, 3, *cfg.eval_crop)  # [1024, 1024]
        # dummy_input = torch.randn(1, 3, 1024, 2048)
        input_names = ['input_image']
        output_names = ['preds',]

        torch.onnx.export(net, dummy_input, args.out_pth,
        input_names=input_names, output_names=output_names,
        verbose=False, opset_version=11, )

