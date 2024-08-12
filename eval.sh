### bev2024
# python tools/evaluate.py \
# --config configs/bisenetv1_bev2024.py \
# --weight-path res/bev_2024/model_final.pth \


### bev2024_1024
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py \
--config configs/bisenetv1_bev20234_1024.py \
--weight-path res/bev_20234_1024_rotate90_2witers/model_final.pth \
--save_pred

### bev2024_1024_6cls
# python tools/evaluate.py \
# --config configs/bisenetv1_bev20234_1024_6cls.py \
# --weight-path res/bev_20234_1024_6cls_2witers/model_final.pth \
# --save_pred


### HYRoad
# python tools/evaluate.py \
# --config configs/bisenetv1_HYRoad.py \
# --weight-path res/HYRoad_5kiters/model_final.pth \
# --save_pred

### cityscapes
# CUDA_VISIBLE_DEVICES=1 python tools/evaluate.py \
# --config configs/bisenetv1_city.py \
# --weight-path res/city_256_512_8witers/model_final.pth \
# --save_pred