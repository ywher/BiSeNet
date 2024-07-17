### bev2024
# python generate_dataset_txt.py \
#     --dataset "bev2024" \
#     --data_path "datasets/bev_2024" \
#     --image_folder "image" \
#     --image_suffix ".png" \
#     --label_folder "label" \
#     --label_suffix ".png" \
#     --split "train"

### bev20234
for split in "train" "val"
do
    python generate_dataset_txt.py \
        --dataset "bev_20234_1024" \
        --data_path "datasets/bev_20234_1024" \
        --image_folder "image" \
        --image_suffix ".png" \
        --label_folder "label" \
        --label_suffix ".png" \
        --split ${split}
done