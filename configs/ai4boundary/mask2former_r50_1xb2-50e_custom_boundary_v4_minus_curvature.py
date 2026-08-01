# ============================================================
# V4 leave-one-out — 去掉曲率正则 loss（Track3 增量）
#
# 其余保持 v4_combined 原样：Track1 argmax(balanced) + Track2 boundary
# loss + surface loss。用于和 v4_combined 对比，验证 Track3 这个"增量"
# 是否真的有正向贡献（plans.md 里已经提示这项预期贡献很小）。
# ============================================================

_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v4_combined.py']

model = dict(
    panoptic_head=dict(
        loss_curvature=None))

work_dir = './work_dirs/mask2former_r50_1xb2-50e_custom_boundary_v4_minus_curvature'
