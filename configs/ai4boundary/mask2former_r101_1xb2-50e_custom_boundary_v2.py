# ============================================================
# Backbone 对照实验 — ResNet101
#
# 固定 V2 的 loss/数据配方，只把 backbone 从 R50 换成 R101（更深、参数更多，
# in_channels 结构不变，仍是 [256,512,1024,2048]）。
# ============================================================

_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v2.py']

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')))

work_dir = './work_dirs/mask2former_r101_1xb2-50e_custom_boundary_v2'

# NOTE: R101 backbone activations are noticeably heavier than R50 at the
# same batch size (see memory estimate in chat). If this OOMs at the
# inherited batch_size=12, lower train_dataloader.batch_size via
# --cfg-options (e.g. to 8) rather than editing this file.
