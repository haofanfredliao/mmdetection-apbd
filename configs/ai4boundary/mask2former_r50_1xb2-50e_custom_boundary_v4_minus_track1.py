# ============================================================
# V4 leave-one-out — 去掉 Track1 的推理后处理（还原成默认 fusion head）
#
# 其余保持 v4_combined 原样：Track2 boundary+surface loss + Track3
# curvature（训练权重不变，因为 argmax/mask-nms 只影响推理，不影响训练）。
# 还原成 v2 继承下来的默认后处理（无 argmax、无 mask-nms、无 score_thr），
# 用于验证 argmax(balanced) 这个后处理选择本身贡献了多少。
# ============================================================

_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v4_combined.py']

model = dict(
    panoptic_fusion_head=dict(type='MaskFormerFusionHead'),
    # _delete_=True: fully replace (not merge) test_cfg, otherwise
    # score_thr/argmax_instance/mask_nms_iou_thr from v4_combined would
    # leak through (score_thr in particular is honored by the base
    # MaskFormerFusionHead too, not just the Field variant).
    test_cfg=dict(
        _delete_=True,
        panoptic_on=False,
        instance_on=True,
        semantic_on=False,
        max_per_image=100,
        iou_thr=0.8,
        filter_low_score=True))

work_dir = './work_dirs/mask2former_r50_1xb2-50e_custom_boundary_v4_minus_track1'
