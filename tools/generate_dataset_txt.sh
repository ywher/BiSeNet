### bev2024
# for split in "train" "val"
# do
#     python generate_dataset_txt.py \
#         --dataset "bev_2024_6cls" \
#         --data_path "datasets/bev_2024_6cls" \
#         --image_folder "image" \
#         --image_suffix ".png" \
#         --label_folder "label" \
#         --label_suffix ".png" \
#         --split "${split}"
# done

### bev20234
# for split in "train" "val"
# do
#     python generate_dataset_txt.py \
#         --dataset "bev_20234_1024" \
#         --data_path "datasets/bev_20234_1024" \
#         --image_folder "image" \
#         --image_suffix ".png" \
#         --label_folder "label" \
#         --label_suffix ".png" \
#         --split ${split}
# done


# for split in "train" "val"
# do
#     python generate_dataset_txt.py \
#         --dataset "bev_20234_1024_6cls" \
#         --data_path "datasets/bev_20234_1024_6cls" \
#         --image_folder "image" \
#         --image_suffix ".png" \
#         --label_folder "label" \
#         --label_suffix ".png" \
#         --split ${split}
# done

### HYRoad
# dataset="HYRoad_3cls"
# for split in "train" "val"
# do
#     python generate_dataset_txt.py \
#         --dataset "${dataset}" \
#         --data_path "datasets/${dataset}" \
#         --image_folder "image" \
#         --image_suffix ".png" \
#         --label_folder "label" \
#         --label_suffix ".png" \
#         --split ${split}
# done


### kyxz
dataset="kyxz"
for split in "train" "val"
do
    python generate_dataset_txt.py \
        --dataset "${dataset}" \
        --data_path "datasets/${dataset}" \
        --image_folder "image" \
        --image_suffix ".png" \
        --label_folder "label" \
        --label_suffix ".png" \
        --split ${split}
done

### cityscapes
# for split in "train_256" "val_256"
# do
#     python generate_dataset_txt.py \
#         --dataset "cityscapes" \
#         --data_path "datasets/cityscapes" \
#         --image_folder "leftImg8bit" \
#         --image_suffix "_leftImg8bit.png" \
#         --label_folder "gtFine" \
#         --label_suffix "_gtFine_labelTrainIds.png" \
#         --split ${split} \
#         --subfolder True
# done

### mapillary
# for split in "train" "val"
# do
#     python generate_dataset_txt.py \
#         --dataset "mapillary" \
#         --data_path "datasets/mapillary" \
#         --image_folder "training/images" \
#         --image_suffix ".png" \
#         --label_folder "label" \
#         --label_suffix ".png" \
#         --split ${split}
# done