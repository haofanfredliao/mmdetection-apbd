# E1 post-processing experiment for Mask2Former V3.
#
# Use this config with the same checkpoint as V3 to test whether simple
# instance filtering reduces over-segmentation and duplicate predictions.

_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v3_clean_bg.py']

model = dict(
    test_cfg=dict(
        panoptic_on=False,
        instance_on=True,
        semantic_on=False,
        max_per_image=100,
        score_thr=0.05,
        mask_nms_iou_thr=0.75))

work_dir = './work_dirs/mask2former_r50_1xb2-50e_custom_boundary_v3_clean_bg_e1_mask_nms'
