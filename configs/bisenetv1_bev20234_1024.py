
cfg = dict(
    model_type='bisenetv1',
    n_cats=7,
    num_aux_heads=2,
    lr_start=1e-2,
    weight_decay=5e-4,
    warmup_iters=1000,
    max_iter=20000,  # 80000
    dataset='Bev',
    im_root='./datasets/bev_20234_1024',
    train_im_anns='./datasets/bev_20234_1024/train.txt',
    val_im_anns='./datasets/bev_20234_1024/val.txt',
    scales=[0.75, 2.],
    cropsize=[1024, 1024],  # [512, 512]
    eval_crop=[1024, 1024],  # [512, 512]
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    ims_per_gpu=8,
    eval_ims_per_gpu=2,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res/bev_20234_1024',
)
