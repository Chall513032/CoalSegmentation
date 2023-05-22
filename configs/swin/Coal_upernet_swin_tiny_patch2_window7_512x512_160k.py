_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/marble.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_320k.py'
]
model = dict(
    backbone=dict(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        patch_norm=True
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=4,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5),
            dict(type='FocalLoss', loss_weight=0.5)]
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=4
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.0005, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=5e-4,
                 power=2.0, min_lr=0.0, by_epoch=False)

# lr_config = dict(_delete_=True, policy='poly',
#                  power=2.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)
