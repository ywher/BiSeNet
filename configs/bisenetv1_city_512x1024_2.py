
cfg = dict(
    model_type='bisenetv1',
    n_cats=19,
    num_aux_heads=2,
    loss_type='OhemCELoss',
    lr_start=1e-2,
    weight_decay=5e-4,
    warmup_iters=1000,
    max_iter=80000,
    dataset='CityScapes',
    color_dataset='cityscapes',
    im_root='./datasets/cityscapes',
    train_im_anns='./datasets/cityscapes/train.txt',
    val_im_anns='./datasets/cityscapes/val.txt',
    scales=[0.75, 2.],
    train_im_scale=0.5,
    cropsize=[1024, 2048],  # (h, w)
    rotate=0,
    eval_crop=[512, 1024],
    eval_scales=[1.0],  # [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    rgb_mean=(0.3257, 0.3690, 0.3223),
    rgb_std=(0.2112, 0.2148, 0.2115),
    ims_per_gpu=8,
    eval_ims_per_gpu=1,
    eval_intervals=1000,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res/city_512_1024_2_8witers',
)
