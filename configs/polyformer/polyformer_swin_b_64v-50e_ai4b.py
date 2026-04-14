# PolyFormer Instance Segmentation — Agricultural Field Boundary Detection
# Based on: "PolyFormer: Referring Image Segmentation as Sequential Polygon
# Generation" (Liu et al., CVPR 2023).
#
# Key motivation for agricultural fields:
#   • Fields have naturally polygonal boundaries — vertex-based representation
#     is more compact and boundary-sharp than pixel masks.
#   • PolyFormer's bilinear coordinate refinement gives sub-pixel accuracy,
#     improving thin boundary IoU compared to Mask2Former's pixel prediction.
#   • The DETR-style parallel decoder enables efficient multi-instance det.
#
# Backbone: Swin-Base (384 pretrain, ImageNet-22K)
# Neck:     FPN  (P2–P5)
# Head:     PolyFormerInsHead  (256-dim, 100 queries, 64 vertices)

_base_ = ['../_base_/default_runtime.py',
          '../_base_/datasets/coco_instance.py',
          '../_base_/schedules/schedule_1x.py']

# ── Dataset ──────────────────────────────────────────────────────────────────
data_root = 'data/data/ai4b_coco/'
metainfo = dict(classes=('field', ), palette=[(220, 20, 60)])
num_classes = 1

# ── Model ────────────────────────────────────────────────────────────────────
model = dict(
    type='PolyFormer',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),

    # Swin-Base backbone (same width as PolyFormer-B visual encoder)
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        # Download from:
        # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/
        #   swin_base_patch4_window12_384_22k.pth
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/swin_base_patch4_window12_384_22k.pth')),

    # FPN: aggregate 4 Swin stages → 256-d feature pyramid
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=4),

    # PolyFormer head
    bbox_head=dict(
        type='PolyFormerInsHead',
        num_classes=num_classes,
        in_channels=[256, 256, 256, 256],  # from FPN
        embed_dims=256,
        num_queries=100,
        num_vertices=64,   # 64 vertices well captures field polygon detail

        encoder_cfg=dict(
            num_layers=6,
            num_heads=8,
            ffn_ratio=4.0,
            dropout=0.1,
            image_bucket_size=42,    # PolyFormer default
            attn_scale_factor=2.0),  # PolyFormer default

        decoder_cfg=dict(
            num_layers=6,
            num_heads=8,
            ffn_ratio=4.0,
            dropout=0.1),

        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            # Background weight = 0.1 following DETR convention
            class_weight=[1.0] * num_classes + [0.1]),

        loss_poly=dict(
            type='L1Loss',
            loss_weight=5.0),

        test_cfg=dict(score_thr=0.5, max_per_img=200)),

    train_cfg=dict(),
    test_cfg=dict())

# ── Data loaders ─────────────────────────────────────────────────────────────
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# ── Evaluators ───────────────────────────────────────────────────────────────
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric=['bbox', 'segm'],
    format_only=False)
test_evaluator = val_evaluator

# ── Optimiser & schedule ─────────────────────────────────────────────────────
# AdamW with layer-decay following ViT/Swin best practice.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=0.05),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'query_feat': dict(lr_mult=1.0),
            'query_pos': dict(lr_mult=1.0),
        }))

# 50 epochs
max_epochs = 50
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs,
                 val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[35, 45],
        gamma=0.1)
]

# ── Misc ─────────────────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3))
