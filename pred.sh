### bev_20234
# CUDA_VISIBLE_DEVICES=0 python tools/predict.py \
# --config "configs/bisenetv1_bev20234_1024.py" \
# --weight_path "res/bev_20234_1024_4witers/model_final.pth" \
# --image_folder "datasets/bev_20234_1024/1-L" \
# --output_folder "res/bev_20234_1024_4witers/pred_1-L" \
# --dataset "bev_20234" \

### HYRoad
# work_root="res/HYRoad_2witers"
# CUDA_VISIBLE_DEVICES=0 python tools/predict.py \
# --config "configs/bisenetv1_HYRoad.py" \
# --weight_path "${work_root}/model_final.pth" \
# --image_folder "datasets/HYRoad/all_images" \
# --output_folder "${work_root}/pred_all" \
# --dataset "HYRoad" \


work_root="res/city_1024_8witers"
CUDA_VISIBLE_DEVICES=0 python tools/predict.py \
--config "configs/bisenetv1_city.py" \
--weight_path "${work_root}/model_final.pth" \
--image_folder "datasets/iphone" \
--output_folder "${work_root}/pred_iphone" \
--dataset "cityscapes" \
--height 1024