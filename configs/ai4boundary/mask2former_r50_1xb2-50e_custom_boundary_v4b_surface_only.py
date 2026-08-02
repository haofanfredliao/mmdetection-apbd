# ============================================================
# 消融实验 — 仅 Track2 的 Kervadec surface loss（单变量）
#
# 基于 v3_clean_bg（含继承的 BoundaryDiceLoss + 干净数据配方），只加
# loss_surface，推理端保持 v3 默认（不做 argmax/mask-nms）。
#
# Track3 已废弃（原因见 v4_combined 配置头部），而 Track1 是纯推理期改动，
# 训练图与本配置完全相同 —— 所以这一个训练任务同时覆盖了 v4_combined：
# 训完之后拿同一份权重换 test_cfg 评估即可得到 "+Track1" 那一格，
# 不需要单独再训一次。save_best 用未经 argmax 削弱的 mAP 口径挑权重。
#
# surface loss 的权重与 ramp 已按首轮 v4 的诊断结论修正，
# 见 v4_combined 配置里的说明。
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
        end_weight=0.5,
        begin_iter=0,
        end_iter=int(max_iters * 0.1)),
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
