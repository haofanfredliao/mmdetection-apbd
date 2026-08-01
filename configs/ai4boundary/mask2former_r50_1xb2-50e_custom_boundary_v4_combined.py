# ============================================================
# Mask2Former R50 – AI4Boundary – V4 "combined" (kitchen-sink) experiment
#
# Stacks every track from docs/plans.md that has shown promise so far,
# instead of testing them one at a time:
#
#   Track 1 (E1, inference-only post-processing):
#     argmax instance assignment, "balanced" preset — every pixel is
#     assigned to at most one instance (structurally non-overlapping),
#     with score_thr=0.2 / iou_thr=0.7 chosen to trade off recall vs.
#     duplicate suppression. Same settings as
#     mask2former_r50_1xb2-50e_custom_boundary_v2_e1b_argmax_balanced.py.
#     No training-time change; only test_cfg + fusion head differ.
#
#   Track 2 (E4, train-time boundary supervision):
#     - loss_boundary: existing BoundaryDiceLoss, inherited unchanged from
#       V2 (morphological dilation/erosion boundary-band Dice).
#     - loss_surface: NEW KervadecBoundaryLoss ("surface loss" in the
#       original Kervadec et al. preprint) — GT signed-distance-map loss
#       for direct boundary localization, complementing the region-based
#       Dice/CE + boundary-band losses above.
#
#   Track 3 (E5, train-time, "increment"):
#     - loss_curvature: NEW CurvatureLoss, penalizes jagged predicted
#       boundaries via a hinge on estimated curvature magnitude, with a
#       dead-zone (tau) that protects genuine sharp corners (e.g. the
#       near-90° corners typical of field boundaries) from being
#       smoothed away. Deliberately kept low-weight per plans.md's note
#       that this is meant to be a small increment on top of Track 1/2,
#       not a primary driver.
#
# Parameter adjustments (this file's whole reason for existing beyond
# stacking losses):
#   - Fixes a checkpoint-frequency bug inherited from the upstream
#     mask2former_r50_8xb2-lsj-50e_coco-panoptic.py base config:
#     CheckpointHook there is by_epoch=False (iteration-based), but every
#     downstream ai4boundary_v1/v2/v3 config sets interval=5 assuming
#     "5 epochs" — in reality this saved a checkpoint every 5
#     ITERATIONS (confirmed by work_dirs/..._v3_clean_bg containing
#     iter_16490/16495/16500.pth), flooding disk I/O and slowing training.
#   - Eval + checkpoint intervals are aligned (checkpoint fires exactly at
#     each eval) and both run every ~2 epochs' worth of iterations
#     instead of every ~1, since this run does more per-iteration work
#     (4 aux losses) and doesn't need epoch-level eval granularity.
# ============================================================

_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v3_clean_bg.py']

model = dict(
    panoptic_head=dict(
        # Track 2 addition: Kervadec surface loss alongside the existing
        # (inherited) BoundaryDiceLoss. Start small — its gradient scales
        # with the GT distance map, which is large/unstable before the
        # mask has roughly converged — and ramp up via the
        # LossWeightRampHook below (mirrors the original paper's alpha
        # schedule).
        loss_surface=dict(
            type='KervadecBoundaryLoss',
            loss_weight=0.001,
            max_distance=32),
        surface_max_res=128,
        # Track 3 "increment": curvature regularization, low weight.
        loss_curvature=dict(
            type='CurvatureLoss',
            loss_weight=0.05,
            tau=0.3,
            band_kernel_size=5),
        curvature_max_res=128),
    # Track 1: switch to the argmax/mask-nms capable fusion head.
    panoptic_fusion_head=dict(type='FieldMaskFormerFusionHead'),
    test_cfg=dict(
        panoptic_on=False,
        instance_on=True,
        semantic_on=False,
        max_per_image=100,
        score_thr=0.2,
        argmax_instance=True,
        iou_thr=0.7,
        filter_low_score=False))

# Same total optimization budget as V3 clean_bg for a fair comparison
# (docs/plans.md "实验规范": align max_iters/LR schedule across configs).
max_iters = 16500

custom_hooks = [
    dict(
        type='LossWeightRampHook',
        module_path='panoptic_head.loss_surface',
        start_weight=0.001,
        end_weight=0.05,
        begin_iter=0,
        end_iter=int(max_iters * 0.3)),
]

# ---------------- eval / checkpoint frequency ----------------
# V3 clean_bg has ~330 iters/epoch; eval+checkpoint every 2 epochs instead
# of every 1 to cut I/O and eval overhead on this heavier (4-loss) run.
val_interval_iters = 660

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
        by_epoch=False,
        interval=val_interval_iters,  # aligned with eval; was a buggy 5
        max_keep_ckpts=3,
        save_best='coco/segm_mAP',
        rule='greater'))

work_dir = './work_dirs/mask2former_r50_1xb2-50e_custom_boundary_v4_combined'
