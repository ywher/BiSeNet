
cfg = dict(
    model_type='bisenetv1',
    n_cats=7,
    num_aux_heads=2,
    lr_start=1e-2,
    weight_decay=5e-4,
    warmup_iters=1000,
    max_iter=10000,  # 80000
    dataset='Bev',
    color_dataset='bev_20234',
    im_root='./datasets/bev_20234_1024',
    train_im_anns='./datasets/bev_20234_1024/train.txt',
    val_im_anns='./datasets/bev_20234_1024/val.txt',
    scales=[0.75, 2.],
    cropsize=[1024, 1024],  # [512, 512]
    eval_crop=[1024, 1024],  # [512, 512]
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    rgb_mean=(0.3755, 0.3719, 0.3771),
    rgb_std=(0.1465, 0.1529, 0.1504), 
    ims_per_gpu=8,
    eval_ims_per_gpu=2,  # 2
    eval_intervals=2000,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res/bev_20234_1024_1witers',
)
