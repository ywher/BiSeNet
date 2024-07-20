
### 
exp_root="res/bev_20234_1024_2witers"
python tools/export_onnx.py \
--config configs/bisenetv1_bev20234_1024.py \
--weight-path ${exp_root}/model_final.pth \
--outpath ${exp_root}/model.onnx \
--aux-mode eval