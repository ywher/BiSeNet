# ### bev2024
# data_path="datasets/bev_2024"
# split="train"

### bev20234
# data_path="datasets/bev_20234_1024"
# split="train"

# python check_dataset_info.py \
# data_path="datasets/bev_20234_1024_6cls"
# split="train"

### HYRoad
# data_path="datasets/HYRoad"
# split="train"

### cityscapes
data_path="datasets/cityscapes"
split="val"


python check_dataset_info.py \
--im_root ../${data_path} \
--im_anns ../${data_path}/${split}.txt