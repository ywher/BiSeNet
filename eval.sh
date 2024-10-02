### bev2024
# python tools/evaluate.py \
# --config configs/bisenetv1_bev2024.py \
# --weight-path res/bev_2024/model_final.pth \


### bev2024_1024
# CUDA_VISIBLE_DEVICES=1 python tools/evaluate.py \
# --config configs/bisenetv1_bev20234_1024.py \
# --weight-path res/bev_20234_1024_rotate45_4witers/model_final.pth \
# --save_pred

### bev2024_1024_6cls
# python tools/evaluate.py \
# --config configs/bisenetv1_bev20234_1024_6cls.py \
# --weight-path res/bev_20234_1024_6cls_rotate90_2witers/model_final.pth \
# --save_pred


### HYRoad
# python tools/evaluate.py \
# --config configs/bisenetv1_HYRoad.py \
# --weight-path res/HYRoad_5kiters/model_final.pth \
# --save_pred

### cityscapes
# config=bisenetv1_city_512x1024
# config=bisenetv1_city_512x1024_2
config=bisenetv1_city_1024x2048
# res_folder=city_512_1024_8witers
# res_folder=city_512_1024_2_8witers
res_folder=city_1024_2048_8witers
CUDA_VISIBLE_DEVICES=1 python tools/evaluate.py \
--config configs/${config}.py \
--weight-path res/${res_folder}/model_final.pth \
--save_pred \
--test_time \
# --up_scale 2.0