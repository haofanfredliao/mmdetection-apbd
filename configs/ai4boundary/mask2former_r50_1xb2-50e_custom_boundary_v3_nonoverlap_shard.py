_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v3_nonoverlap.py']

# Shard-safe full training.
# Original V2 used batch_size=12 for 20000 iters.  On GPU shards we use
# batch_size=2 with gradient accumulation x6, and scale iteration counts by 6
# so the run still sees roughly the same number of training samples and
# optimizer steps.
train_dataloader = dict(
    batch_size=2,
    num_workers=4)

optim_wrapper = dict(accumulative_counts=6)

max_iters = 120000
val_interval_iters = 2400  # ~1 epoch with batch_size=2 after filtering.

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

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=12000,
        max_keep_ckpts=3,
        save_best='coco/segm_mAP',
        rule='greater'))
