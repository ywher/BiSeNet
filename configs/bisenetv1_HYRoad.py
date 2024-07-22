
cfg = dict(
    model_type='bisenetv1',
    n_cats=15,
    num_aux_heads=2,
    lr_start=1e-2,
    weight_decay=5e-4,
    warmup_iters=500,
    max_iter=5000,  # 80000
    dataset='HYRoad',
    color_dataset='HYRoad',
    im_root='./datasets/HYRoad',
    train_im_anns='./datasets/HYRoad/train.txt',
    val_im_anns='./datasets/HYRoad/val.txt',
    scales=[0.75, 2.],
    cropsize=[1024, 1024],  # [512, 512]
    eval_crop=[1024, 1024],  # [512, 512]
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    rgb_mean=(0.4753, 0.4882, 0.4546),
    rgb_std=(0.1994, 0.2052, 0.2469),
    ims_per_gpu=8,
    eval_ims_per_gpu=2,
    eval_intervals=500,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res/HYRoad_2witers',
)
