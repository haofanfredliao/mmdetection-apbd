# ============================================================
# Backbone comparison — ResNet101 on the V5 recipe
#
# Holds the entire V5 recipe fixed (BoundaryDiceLoss + NonOverlapLoss +
# point-sampled KervadecBoundaryLoss, both ramps, argmax fusion head and its
# test_cfg) and swaps only the backbone. R101 keeps ResNet's channel layout
# [256, 512, 1024, 2048], so panoptic_head.in_channels needs no change.
#
# Batch size 12 fits on the H800 (~65 GB peak in a probe forward+backward);
# no gradient accumulation needed. Schedule inherits from R50-V5 unchanged.
# ============================================================

_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v5.py']

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./pretrain/resnet101-63fe2227.pth')))

work_dir = './work_dirs/mask2former_r101_1xb2-50e_custom_boundary_v5'
