# ============================================================
# Mask2Former R50 – AI4Boundary – Boundary Aux Loss V3
#
# Changes vs V2:
#   1. Uses the corrected COCO conversion where all-zero instance-channel
#      tiles are kept as negative images with zero annotations.
#   2. Trains on all 1_good samples plus a deterministic 20% background
#      negative sample, excluding non-empty 2_lazy and all 3_extreme samples.
#   3. Evaluates on 1_good only, keeping COCO + FieldSegmentationMetric
#      aligned with clean positive instance quality.
# ============================================================

_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v2.py']

background_sample_ratio = 0.2
background_sample_seed = 42

train_dataloader = dict(
    dataset=dict(
        filter_mode='good_with_background',
        filter_cfg=None,
        background_sample_ratio=background_sample_ratio,
        background_sample_seed=background_sample_seed,
        lazy_loss_weight=1.0))

val_dataloader = dict(
    dataset=dict(
        filter_mode='good_only',
        lazy_loss_weight=1.0))

test_dataloader = dict(
    dataset=dict(
        filter_mode='good_only',
        lazy_loss_weight=1.0))

work_dir = './work_dirs/mask2former_r50_1xb2-50e_custom_boundary_v3_clean_bg'

# Clean train size: 3293 good + round(3293 * 0.2) background = 3952.
# With batch_size=12 this is ~330 iters/epoch; keep the V2 50-epoch budget.
max_iters = 16500
val_interval_iters = 330

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
