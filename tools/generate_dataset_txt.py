import os
import os.path as osp
import argparse
import tqdm
import random

os.chdir(osp.abspath(osp.dirname(osp.dirname(__file__))))

def get_argparse():
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='bev2024', help='dataset name')
    args.add_argument('--data_path', type=str, default='datasets/bev2024', help='dataset path')
    args.add_argument('--image_folder', type=str, default='image', help='image folder')
    args.add_argument('--image_suffix', type=str, default='.png', help='image suffix')
    args.add_argument('--label_folder', type=str, default='label', help='label folder')
    args.add_argument('--label_suffix', type=str, default='.png', help='label suffix')
    args.add_argument('--split', type=str, default='train', help='train list')
    args.add_argument('--subfolder', type=bool, default=False, help='whether the images and labels are in subfolders')
    
    return args.parse_args()

def generate_dataset_txt(dataset, data_path, image_folder, image_suffix, label_folder, label_suffix, split):
    print("Current working directory:", os.getcwd())
    print("File path:", __file__)
    work_root = osp.abspath(osp.dirname(osp.dirname(__file__)))
    print(f'work root: {work_root}')
    image_abs_path = osp.join(work_root, data_path, image_folder, split)
    label_abs_path = osp.join(work_root, data_path, label_folder, split)
    
    image_names = os.listdir(image_abs_path)
    label_names = os.listdir(label_abs_path)
    
    image_names = [name for name in image_names if name.endswith(image_suffix)]
    label_names = [name for name in label_names if name.endswith(label_suffix)]
    
    image_names.sort()
    label_names.sort()
    
    image_pathes = [osp.join(image_folder, split, name) for name in image_names]
    label_pathes = [osp.join(label_folder, split, name) for name in label_names]
    
    bar = tqdm.tqdm(total=len(image_pathes))
    with open(osp.join(data_path, f'{split}.txt'), 'w') as f:
        for image_path, label_path in zip(image_pathes, label_pathes):
            f.write(f'{image_path},{label_path}\n')
            bar.update(1)
    bar.close()
            
    print(f'{split} txt file has been generated.')
    
def generate_dataset_txt_subfolder(dataset, data_path, image_folder, image_suffix, label_folder, label_suffix, split):
    print("Current working directory:", os.getcwd())
    print("File path:", __file__)
    work_root = osp.abspath(osp.dirname(osp.dirname(__file__)))
    print(f'work root: {work_root}')
    
    subfolders = os.listdir(osp.join(work_root, data_path, image_folder, split))
    image_pathes = []
    label_pathes = []
    for subfolder in subfolders:
        image_abs_path = osp.join(work_root, data_path, image_folder, split, subfolder)
        label_abs_path = osp.join(work_root, data_path, label_folder, split, subfolder)
    
        image_names = os.listdir(image_abs_path)
        label_names = os.listdir(label_abs_path)
        
        image_names = [name for name in image_names if name.endswith(image_suffix)]
        label_names = [name for name in label_names if name.endswith(label_suffix)]
        
        image_names.sort()
        label_names.sort()
        
        image_pathes.extend([osp.join(image_folder, split, subfolder, name) for name in image_names])
        label_pathes.extend([osp.join(label_folder, split, subfolder, name) for name in label_names])
    
    # rand the image and label pathes jointly
    rand_idx = list(range(len(image_pathes)))

    random.shuffle(rand_idx)
    image_pathes = [image_pathes[idx] for idx in rand_idx]
    label_pathes = [label_pathes[idx] for idx in rand_idx]
    
    bar = tqdm.tqdm(total=len(image_pathes))
    with open(osp.join(data_path, f'{split}.txt'), 'w') as f:
        for image_path, label_path in zip(image_pathes, label_pathes):
            f.write(f'{image_path},{label_path}\n')
            bar.update(1)
    bar.close()
            
    print(f'{split} txt file has been generated.')
    
if __name__ == '__main__':
    args = get_argparse()
    if args.subfolder:
        generate_dataset_txt_subfolder(args.dataset, args.data_path, args.image_folder, args.image_suffix, args.label_folder, args.label_suffix, args.split)
    else:
        generate_dataset_txt(args.dataset, args.data_path, args.image_folder, args.image_suffix, args.label_folder, args.label_suffix, args.split)