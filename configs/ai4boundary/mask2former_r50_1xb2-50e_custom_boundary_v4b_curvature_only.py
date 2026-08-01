# ============================================================
# 消融实验 — 仅 Track3 的曲率正则 loss（单变量，不叠加 Track1/Track2 新增项）
#
# 基于 v3_clean_bg，只加 loss_curvature，推理端保持默认。用于独立衡量
# "边界规整"这个增量单独的贡献。
# ============================================================

_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v3_clean_bg.py']

model = dict(
    panoptic_head=dict(
        loss_curvature=dict(
            type='CurvatureLoss',
            loss_weight=0.05,
            tau=0.3,
            band_kernel_size=5),
        curvature_max_res=128))

max_iters = 16500
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

work_dir = './work_dirs/mask2former_r50_1xb2-50e_custom_boundary_v4b_curvature_only'
