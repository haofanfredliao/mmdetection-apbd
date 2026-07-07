_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v2.py']

# E1b: high-recall argmax assignment.  Lower score threshold and keep all
# assigned pixels to test whether the previous AP drop came from filtering.
model = dict(
    panoptic_fusion_head=dict(type='FieldMaskFormerFusionHead'),
    test_cfg=dict(
        panoptic_on=False,
        instance_on=True,
        semantic_on=False,
        max_per_image=100,
        score_thr=0.1,
        argmax_instance=True,
        iou_thr=0.5,
        filter_low_score=False))
