# ============================================================
# Mask R-CNN R50-FPN baseline — controlled against Mask2Former V5
#
# Same controlled variables as
# mask2former_r50_1xb2-50e_custom_boundary_v5.py:
#   - QualityAwareCocoDataset
#   - train: good_with_background (20% empty negatives, seed 42)
#   - val/test: good_only
#   - evaluators: CocoMetric (bbox+segm) + FieldSegmentationMetric
#   - batch_size=12, max_iters=16500, val every 660 iters
#   - LR milestones at 90% / 95% of schedule
#
# Deliberately left as Mask R-CNN's own recipe (not V5's):
#   - SGD + standard COCO short-side-800 pipeline (not LSJ 1024)
#   - no BoundaryDice / NonOverlap / Surface / argmax fusion
#   This is the architecture/method baseline, not a reimplementation
#   of the inductive-bias stack on Mask R-CNN.
# ============================================================

_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/default_runtime.py',
]

# ---------------- model ----------------
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# ---------------- data (aligned with V5) ----------------
data_root = 'data/data/ai4b_coco/'
quality_csv = data_root + 'quality_report.csv'
metainfo = dict(
    classes=('field', ),
    palette=[(220, 20, 60)])

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

background_sample_ratio = 0.2
background_sample_seed = 42

train_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='QualityAwareCocoDataset',
        quality_csv=quality_csv,
        filter_mode='good_with_background',
        filter_cfg=None,  # keep empty-annotation background tiles
        background_sample_ratio=background_sample_ratio,
        background_sample_seed=background_sample_seed,
        lazy_loss_weight=1.0,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='QualityAwareCocoDataset',
        quality_csv=quality_csv,
        filter_mode='good_only',
        lazy_loss_weight=1.0,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='QualityAwareCocoDataset',
        quality_csv=quality_csv,
        filter_mode='good_only',
        lazy_loss_weight=1.0,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

# ---------------- eval (aligned with V5) ----------------
val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/instances_val.json',
        metric=['bbox', 'segm'],
        format_only=False),
    dict(type='FieldSegmentationMetric', iou_thr=0.5),
]

test_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/instances_test.json',
        metric=['bbox', 'segm'],
        format_only=False),
    dict(type='FieldSegmentationMetric', iou_thr=0.5),
]

# ---------------- schedule (sample budget = V5) ----------------
# V5: 16500 iters × bs12. Same here so both see ~198k samples.
max_iters = 16500
val_interval_iters = 660

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=val_interval_iters)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# SGD is Mask R-CNN's native optimizer. Base 1x lr=0.02 assumes
# effective batch 16; linear-scale to batch 12.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0001))

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
        end=max_iters,
        by_epoch=False,
        milestones=[int(max_iters * 0.9), int(max_iters * 0.95)],
        gamma=0.1),
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=val_interval_iters,
        max_keep_ckpts=3,
        save_best='coco/segm_mAP',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

auto_scale_lr = dict(enable=False, base_batch_size=16)

load_from = './pretrain/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'

work_dir = './work_dirs/mask-rcnn_r50_fpn_50e_ai4b_v5data'
