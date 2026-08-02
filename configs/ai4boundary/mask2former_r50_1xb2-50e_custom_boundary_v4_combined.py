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
#   Track 3 (E5, curvature regularization): DROPPED after the first v4 run.
#     CurvatureLoss's kappa estimator (Sobel divergence of the unit normal
#     field) does not measure geometric curvature at this discretization: a
#     perfectly straight field edge measures |kappa| ~3.1, not ~0, so tau=0.3
#     left 98.4% of band pixels penalized and the corner-protecting dead zone
#     never engaged. Worse, its minimum is not at the GT — measured on real
#     GT at 128px, a perfect prediction scores 11.78 while an over-smoothed
#     blob scores 5.05, i.e. the loss rewards being smoother than ground
#     truth. That matches what the run produced: the term sat flat at ~0.39
#     for all 16500 iters, and against v3_clean_bg the masks came out
#     smoother (vertices@IoU95 32.9->27.3) but less faithful (Boundary-IoU
#     0.540->0.525, Boundary-F_1px 0.446->0.424, segm_mAP 0.347->0.322 at
#     matched post-processing). Reviving Track 3 needs tau ~5-8 and
#     preferably a contour-turning-angle estimator matching the one in
#     FieldSegmentationMetric._curvature_energy.
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
        surface_max_res=128),
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
        # MUST stay True whenever argmax_instance=True. The argmax in
        # FieldMaskFormerFusionHead._argmax_instance_postprocess runs over the
        # query axis only — there is no background row to lose to — so every
        # pixel is claimed by some query and the masks tile the whole image.
        # This line (mask &= mask_prob >= 0.5) is the only thing that carves
        # background back out. The e1b_argmax_balanced preset this config was
        # copied from had it False; measured on the test set with identical
        # weights, flipping it to True improves every metric at once:
        # segm_mAP 0.246->0.298, Boundary-IoU 0.458->0.520,
        # Boundary-F_1px 0.335->0.411, vertices@IoU95 26.6->12.3,
        # curvature energy 0.600->0.373.
        filter_low_score=True))

# Same total optimization budget as V3 clean_bg for a fair comparison
# (docs/plans.md "实验规范": align max_iters/LR schedule across configs).
max_iters = 16500

# The first v4 run measured loss_surface's raw value (weighted value / its
# current weight) at -0.294 from iter ~5000 onward, which is exactly the value
# a *perfect* prediction scores (measured on real GT at the 128px training
# resolution: perfect = -0.2947, all-0.5 prediction = +13.29). So the loss hit
# its optimum a third of the way in and contributed no gradient afterwards,
# while its weighted magnitude (-0.0147 against a total of ~23.5) was 0.06% of
# the objective. Two changes follow from that:
#   - end_weight 0.05 -> 0.5. Bounded by construction, so this cannot blow up:
#     the worst case (an all-0.5 prediction) contributes 13.29 * 0.5 = 6.6
#     against a total of ~66 at iter 50.
#   - ramp end 30% -> 10% of training. The old schedule reached full weight at
#     iter 4950, i.e. right as the loss saturated at ~iter 5000 — it was at
#     full strength only during the window where it had nothing left to say.
custom_hooks = [
    dict(
        type='LossWeightRampHook',
        module_path='panoptic_head.loss_surface',
        start_weight=0.001,
        end_weight=0.5,
        begin_iter=0,
        end_iter=int(max_iters * 0.1)),
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
