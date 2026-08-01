# ============================================================
# V4 leave-one-out — 去掉 Kervadec surface loss
#
# 其余保持 v4_combined 原样：Track1 argmax(balanced) + Track2 boundary
# loss(不含 surface) + Track3 curvature。用于和 v4_combined 对比，看
# surface loss 这一项到底是加分还是拖累。
# ============================================================

_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v4_combined.py']

model = dict(
    panoptic_head=dict(
        loss_surface=None))

# The surface-loss weight ramp hook has nothing to act on once
# loss_surface is disabled above; drop it entirely instead of letting it
# error out on a missing attribute.
custom_hooks = []

work_dir = './work_dirs/mask2former_r50_1xb2-50e_custom_boundary_v4_minus_surface'
