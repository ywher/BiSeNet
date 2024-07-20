python tools/predict.py \
--config "configs/bisenetv1_bev20234_1024.py" \
--weight_path "res/bev_20234_1024_4witers/model_final.pth" \
--image_folder "datasets/bev_20234_1024/image/val" \
--output_folder "res/bev_20234_1024_4witers/pred" \
--dataset "bev_20234" \