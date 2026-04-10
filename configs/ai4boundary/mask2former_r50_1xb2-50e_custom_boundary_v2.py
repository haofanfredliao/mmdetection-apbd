# ============================================================
# Mask2Former R50 – AI4Boundary – Custom Boundary Loss V2
# Changes vs V1:
#   1. loss_dice: restored to standard DiceLoss (BoundaryDiceLoss was a no-op
#      on point-sampled tensors; kept here as-is for backward compat).
#   2. loss_boundary: NEW aux loss using BoundaryDiceLoss on full-resolution
#      upsampled decoder outputs via Mask2FormerHeadV2.
#   3. Dataset: QualityAwareCocoDataset filters 3_extreme, down-weights 2_lazy.
# ============================================================
_base_ = ['../mask2former/mask2former_r50_8xb2-lsj-50e_coco.py']

# ================= 1. 数据集与类别信息配置 =================
data_root = 'data/data/ai4b_coco/'
quality_csv = data_root + 'quality_report.csv'
metainfo = dict(
    classes=('field', ),
    palette=[(220, 20, 60)]
)

# ================= 2. 模型结构配置 =================
num_things_classes = 1
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes

model = dict(
    panoptic_head=dict(
        # Switch to V2 head which supports boundary aux loss + quality weighting
        type='Mask2FormerHeadV2',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        # Restore standard DiceLoss (BoundaryDiceLoss does not work on
        # point-sampled 2-D tensors – see V1 analysis)
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1]),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        # NEW: boundary aux loss applied ONLY on the last decoder layer,
        # at decoder feature map resolution (≤256×256, no upsample to 1024).
        # This prevents 9× memory overhead of the original naive approach.
        loss_boundary=dict(
            type='BoundaryDiceLoss',
            loss_weight=2.0,
            kernel_size=3,
            eps=1e-5),
        # Cap boundary computation resolution (pred is ~256×256 already;
        # GT is downsampled to this size instead of staying at 1024×1024).
        boundary_max_res=256,
    ),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
    test_cfg=dict(panoptic_on=False, instance_on=True, semantic_on=False)
)

# ================= 3. Pipeline: propagate loss_weight through img_meta =================
# PackDetInputs must list 'loss_weight' so it ends up in img_meta and is
# accessible inside Mask2FormerHeadV2._loss_by_feat_single.
image_size = (1024, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'loss_weight')),
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

# ================= 4. 数据加载器配置 (H100 80GB) =================
train_dataloader = dict(
    batch_size=12,
    num_workers=8,
    dataset=dict(
        type='QualityAwareCocoDataset',
        quality_csv=quality_csv,
        lazy_loss_weight=0.2,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='QualityAwareCocoDataset',
        quality_csv=quality_csv,
        lazy_loss_weight=0.2,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='QualityAwareCocoDataset',
        quality_csv=quality_csv,
        lazy_loss_weight=0.2,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test/'),
        pipeline=test_pipeline))

# ================= 5. 评估器配置 =================
val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/instances_val.json',
        metric=['bbox', 'segm'],
        format_only=False),
    dict(
        type='FieldSegmentationMetric',
        iou_thr=0.5)
]

test_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/instances_test.json',
        metric=['bbox', 'segm'],
        format_only=False),
    dict(
        type='FieldSegmentationMetric',
        iou_thr=0.5)
]

# ================= 6. 优化器与训练策略 =================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00005,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=3,
        save_best='coco/segm_mAP',
        rule='greater'))

# Epoch size shrinks because 3_extreme (~9.8%) is excluded from train set.
# Approximate train size after filtering: 5319 - 524 ≈ 4795 images.
# With batch_size=12: ~400 iters/epoch → 50 epochs ≈ 20000 iters.
max_iters = 20000
val_interval_iters = 400  # ~1 epoch

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=val_interval_iters)

param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[int(max_iters * 0.9), int(max_iters * 0.95)],
    gamma=0.1)
