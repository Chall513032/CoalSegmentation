_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/marble.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_320k.py'
]
model = dict(
    decode_head=dict(
        num_classes=4,
        concat_input=True,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5),
            dict(type='FocalLoss', loss_weight=0.5)]
    ),
    test_cfg=dict(mode='whole')
)

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
evaluation = dict(metric='mDice')
