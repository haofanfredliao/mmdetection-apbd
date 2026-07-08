#!/usr/bin/env python3
"""Run E1 inference-side post-processing sweep for Mask2Former boundary v2."""

import eval_compare


V2_CHECKPOINT = (
    'work_dirs/mask2former_r50_1xb2-50e_custom_boundary_v2/'
    'best_coco_segm_mAP_iter_19600.pth')

E1_MODELS = [
    {
        'name': 'boundary_v2_default',
        'label': 'V2 default instance',
        'config':
        'configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v2.py',
        'checkpoint': V2_CHECKPOINT,
    },
    {
        'name': 'boundary_v2_score03',
        'label': 'V2 score_thr=0.3',
        'config':
        'configs/ai4boundary/'
        'mask2former_r50_1xb2-50e_custom_boundary_v2_e1_score03.py',
        'checkpoint': V2_CHECKPOINT,
    },
    {
        'name': 'boundary_v2_score05',
        'label': 'V2 score_thr=0.5',
        'config':
        'configs/ai4boundary/'
        'mask2former_r50_1xb2-50e_custom_boundary_v2_e1_score05.py',
        'checkpoint': V2_CHECKPOINT,
    },
    {
        'name': 'boundary_v2_masknms075',
        'label': 'V2 score0.3 + maskNMS0.75',
        'config':
        'configs/ai4boundary/'
        'mask2former_r50_1xb2-50e_custom_boundary_v2_e1_masknms075.py',
        'checkpoint': V2_CHECKPOINT,
    },
    {
        'name': 'boundary_v2_argmax',
        'label': 'V2 argmax non-overlap',
        'config':
        'configs/ai4boundary/'
        'mask2former_r50_1xb2-50e_custom_boundary_v2_e1_argmax.py',
        'checkpoint': V2_CHECKPOINT,
    },
]


if __name__ == '__main__':
    eval_compare.MODELS = E1_MODELS
    eval_compare.main()
