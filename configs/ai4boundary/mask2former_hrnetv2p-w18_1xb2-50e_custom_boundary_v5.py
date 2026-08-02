# ============================================================
# Backbone comparison — HRNetV2p-W18 on the V5 recipe
#
# Holds the entire V5 recipe fixed and swaps only the backbone (+ HRFPN
# neck that turns HRNet's 4-branch output into the 4-level same-channel
# pyramid Mask2Former's pixel_decoder expects). Schedule matches R50-V5
# (16500 iters, val every 660).
#
# HRNet keeps high-resolution branches alive through all 4 stages, so it
# can be more memory-hungry than R50 at the same batch size despite fewer
# parameters. If this OOMs, drop train_dataloader.batch_size and scale
# max_iters / ramps the same way R101-V5 does.
# ============================================================

_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v5.py']

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
            type='Pretrained',
            checkpoint='./pretrain/hrnetv2_w18-00eb2006.pth')),
    neck=dict(
        type='HRFPN',
        in_channels=[18, 36, 72, 144],
        out_channels=256,
        # Mask2Former's pixel_decoder wants exactly 4 pyramid levels
        # (strides 4/8/16/32); HRFPN defaults to 5 (adds stride-64).
        num_outs=4),
    panoptic_head=dict(in_channels=[256, 256, 256, 256]))

work_dir = './work_dirs/mask2former_hrnetv2p-w18_1xb2-50e_custom_boundary_v5'
