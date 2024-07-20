
cfg = dict(
    model_type='bisenetv1',
    n_cats=6,
    num_aux_heads=2,
    lr_start=1e-2,
    weight_decay=5e-4,
    warmup_iters=1000,
    max_iter=20000,  # 80000
    dataset='Bev',
    color_dataset='bev_20234_6cls',
    im_root='./datasets/bev_2024_6cls',
    train_im_anns='./datasets/bev_20234_1024_6cls/train.txt',
    val_im_anns='./datasets/bev_2024_6cls/val.txt',
    scales=[0.75, 2.],
    cropsize=[1024, 1024],  # [512, 512]
    eval_crop=[1024, 1024],  # [512, 512]
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    rgb_mean=(0.3697758483886719, 0.3678707580566406, 0.3706438903808594),
    rgb_std=(0.16299784765402195, 0.16775851555258015, 0.16443658523566113), 
    ims_per_gpu=8,
    eval_ims_per_gpu=2,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res/bev_20234_1024_6cls_2witers',
)
