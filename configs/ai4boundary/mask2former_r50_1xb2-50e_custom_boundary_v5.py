# ============================================================
# Mask2Former R50 – AI4Boundary – V5
#
# The configuration the thesis reports as the full model. Each of the three
# inductive biases from docs/plans.md enters exactly once, at the stage where
# it is cheapest to enforce, and nothing else from the v4 kitchen-sink run
# survives:
#
#   Track 1 — planar partition (topology).
#     Training: NonOverlapLoss on the matched positive masks, penalizing
#       sum_i p_i(x) > 1 so that the queries are pushed to claim disjoint
#       regions rather than to each cover the same parcel.
#     Inference: argmax instance assignment in FieldMaskFormerFusionHead,
#       which makes disjointness structural rather than merely encouraged.
#     The loss and the post-processing are complementary, not redundant:
#       argmax alone yields a hard partition of whatever the network
#       produces, but it has to break ties between queries that were never
#       trained to disagree, and those ties are where the spurious slivers
#       come from.
#
#   Track 2 — boundary adherence (geometry).
#     BoundaryDiceLoss (inherited unchanged from v2) for the boundary band,
#     plus KervadecBoundaryLoss with surface_mode='point'. See the note on
#     the point-sampled path below.
#
#   Track 3 — boundary regularity (geometry).
#     Post-processing polygon simplification only; no training-time term.
#     CurvatureLoss was dropped after v4 — its Sobel-divergence estimator
#     scores a perfectly straight field edge at |kappa| ~3.1 instead of ~0,
#     so tau=0.3 penalized 98.4% of band pixels and the corner-protecting
#     dead zone never engaged. Worse, its optimum is not at the GT: on real
#     GT at 128px a perfect prediction scores 11.78 while an over-smoothed
#     blob scores 5.05, i.e. it rewards being smoother than ground truth.
#     The v4 run behaved accordingly — smoother masks (vertices@IoU95
#     32.9->27.3) that fit worse (Boundary-IoU 0.540->0.525, segm_mAP
#     0.347->0.322 at matched post-processing).
#
# Why the surface loss is point-sampled here (surface_mode='point'):
#   In v4 it was averaged densely over the whole 128px frame at the last
#   decoder layer only, and it did nothing. Measured raw value saturated at
#   -0.294 by iter ~5000 — exactly the score of a *perfect* prediction at
#   that resolution — and even after raising the weight 10x to 0.5 it was
#   0.4% of the objective. The dense average is dominated by pixels deep
#   inside or outside the mask, where sigmoid(z) is saturated and therefore
#   contributes value but essentially no gradient. Reading the same signed
#   distance map at the coordinates Mask2Former already importance-samples
#   for dice/CE puts ~75% of the samples in the uncertain band around the
#   predicted contour, which is where d/dz sigmoid(z) is largest and where
#   the sign of phi actually says which way to move the boundary. It also
#   makes the term available to all 9 auxiliary decoder layers instead of
#   the last one alone.
#   Consequence for reading the logs: the *reported value* of loss_surface
#   is now smaller than the dense version's, not larger, because the band
#   it samples has |phi| ~ 0 by construction. Judge it by its gradient
#   contribution, not its magnitude.
#
# How the auxiliary weights below were chosen:
#   By gradient norm, not by loss value — sizing v4's surface weight off its
#   printed value is what produced an inert term. scripts/probe_loss_gradients
#   backprops each term separately from a shared forward pass on the v3
#   checkpoint and reports its gradient norm over the trainable parameters.
#   Every auxiliary term here is set to ~10% of the summed loss_dice gradient
#   norm (74.5 at batch 4), which is where the inherited loss_boundary weight
#   of 2.0 already sat (9.5%), so the three terms share one budget:
#     loss_boundary    w=2.0    -> 9.5% of dice   (inherited, unchanged)
#     loss_nonoverlap  w=6.5    -> 10%  of dice   (was 3.1% at w=2.0)
#     loss_surface     w=0.09   -> 10%  of dice   (84.4 grad norm per unit w)
#   Measured on a checkpoint rather than a random init on purpose: these
#   terms are meant to act on an already roughly-correct mask, and their
#   relative gradient scale is different before that.
# ============================================================

_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v3_clean_bg.py']

max_iters = 16500

model = dict(
    panoptic_head=dict(
        # Track 1, training side. Ramped like the surface term below — see
        # custom_hooks; loss_weight here is only the pre-ramp starting value.
        loss_nonoverlap=dict(
            type='NonOverlapLoss',
            loss_weight=0.0,
            mode='sum_excess',
            power=2.0),
        nonoverlap_max_res=128,
        # Track 2, surface term. Starts near zero and ramps up (see
        # custom_hooks): its gradient scales with the GT distance map, which
        # is large and uninformative while the mask is still far from the GT.
        loss_surface=dict(
            type='KervadecBoundaryLoss',
            loss_weight=0.001,
            max_distance=32),
        surface_max_res=128,
        surface_mode='point'),
    # Track 1, inference side.
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
        # This flag (mask &= mask_prob >= 0.5) is the only thing that carves
        # background back out. Measured on the test set with identical v4
        # weights, flipping it to True improves every metric at once:
        # segm_mAP 0.246->0.298, Boundary-IoU 0.458->0.520,
        # Boundary-F_1px 0.335->0.411, vertices@IoU95 26.6->12.3,
        # curvature energy 0.600->0.373.
        filter_low_score=True))

# Ramp end at 10% of training rather than v4's 30%: the old schedule reached
# full weight at iter 4950, right as the dense loss saturated, so it was at
# full strength only during the window where it had nothing left to say.
#
# end_weight is 0.09, NOT the 0.5 that v4 was heading towards. Sizing it by
# gradient rather than by printed value reverses the direction of the fix —
# see the calibration note above; at 0.5 the point-sampled term would carry
# 57% of the dice gradient, which is not an auxiliary term any more.
#
# loss_nonoverlap is ramped on the same schedule, for a different reason. At
# initialization every matched query predicts ~0.5 everywhere, so with ~12
# instances per image sum_i p_i ~ 6 and the squared excess starts at ~4 raw
# (26.7 weighted, 26% of the total objective, driving grad_norm to 3318 vs a
# steady-state ~150). It self-corrects within ~60 iterations, but the cheapest
# way to satisfy the constraint while the masks are still noise is to predict
# nothing anywhere, and there is no reason to let that gradient near the model
# before the masks mean something.
custom_hooks = [
    dict(
        type='LossWeightRampHook',
        module_path='panoptic_head.loss_surface',
        start_weight=0.001,
        end_weight=0.09,
        begin_iter=0,
        end_iter=int(max_iters * 0.1)),
    dict(
        type='LossWeightRampHook',
        module_path='panoptic_head.loss_nonoverlap',
        start_weight=0.0,
        end_weight=6.5,
        begin_iter=0,
        end_iter=int(max_iters * 0.1)),
]

# ---------------- eval / checkpoint frequency ----------------
# ~330 iters/epoch; eval+checkpoint every 2 epochs to cut I/O and eval
# overhead on this heavier run.
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
        interval=val_interval_iters,
        max_keep_ckpts=3,
        save_best='coco/segm_mAP',
        rule='greater'))

work_dir = './work_dirs/mask2former_r50_1xb2-50e_custom_boundary_v5'
