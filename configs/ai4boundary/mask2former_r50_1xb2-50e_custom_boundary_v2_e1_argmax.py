_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v2.py']

# E1: argmax assignment gives every pixel to at most one instance, matching the
# non-overlap prior at inference time without changing training.
model = dict(
    panoptic_fusion_head=dict(type='FieldMaskFormerFusionHead'),
    test_cfg=dict(
        panoptic_on=False,
        instance_on=True,
        semantic_on=False,
        max_per_image=100,
        score_thr=0.3,
        argmax_instance=True,
        iou_thr=0.8,
        filter_low_score=True))
