# ============================================================
# Backbone 对照实验 — HRNetV2p-W18
#
# 固定 V2 的 loss/数据配方，只替换 backbone。HRNet 原生输出 4 个分支
# （不同分辨率、不同通道数，非 ResNet 式金字塔），需要接一个 HRFPN neck
# 把它转成 Mask2Former pixel_decoder 期望的 4 级同通道数金字塔
# （strides=[4,8,16,32]，与 ResNet 版一致，无需改 panoptic_head.strides）。
# HRNet 在这个代码库里此前只接入过 Mask R-CNN/Cascade，这是第一次接入
# Mask2Former，属于新集成，训练前务必先跑冒烟测试。
# ============================================================

_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v2.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144))),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w18')),
    neck=dict(
        type='HRFPN',
        in_channels=[18, 36, 72, 144],
        out_channels=256,
        # Mask2Former's pixel_decoder wants exactly 4 pyramid levels
        # (strides 4/8/16/32); HRFPN defaults to 5 (adds stride-64), so
        # this must be set explicitly.
        num_outs=4),
    panoptic_head=dict(in_channels=[256, 256, 256, 256]))

# ---------------- fix inherited checkpoint-interval bug ----------------
# See mask2former_swin-t_..._v2.py for why this override is necessary.
val_interval_iters = 800

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=20000,
    val_interval=val_interval_iters)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=val_interval_iters,
        max_keep_ckpts=3,
        save_best='coco/segm_mAP',
        rule='greater'))

work_dir = './work_dirs/mask2former_hrnetv2p-w18_1xb2-50e_custom_boundary_v2'

# NOTE: HRNet keeps high-resolution branches alive through all 4 stages,
# so it can be more memory-hungry than ResNet50 at the same batch size
# despite fewer channels. Lower train_dataloader.batch_size via
# --cfg-options if this OOMs at the inherited batch_size=12.
