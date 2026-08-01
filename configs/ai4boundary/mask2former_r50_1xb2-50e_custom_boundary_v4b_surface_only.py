# ============================================================
# 消融实验 — 仅 Track2 的 Kervadec surface loss（单变量，不叠加 Track1/Track3）
#
# 基于 v3_clean_bg（含继承的 BoundaryDiceLoss + 干净数据配方），只加
# loss_surface，推理端保持默认（不做 argmax/mask-nms）。用于和
# v4_combined 及其它 leave-one-out 变体对比，独立衡量 surface loss 的
# 贡献，而不与 argmax/曲率正则的效果混在一起。
# ============================================================

_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v3_clean_bg.py']

model = dict(
    panoptic_head=dict(
        loss_surface=dict(
            type='KervadecBoundaryLoss',
            loss_weight=0.001,
            max_distance=32),
        surface_max_res=128))

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

work_dir = './work_dirs/mask2former_r50_1xb2-50e_custom_boundary_v4b_surface_only'
