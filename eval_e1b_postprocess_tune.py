#!/usr/bin/env python3
"""Run a focused E1b sweep to recover AP while keeping OS low."""

import eval_compare


V2_CHECKPOINT = (
    'work_dirs/mask2former_r50_1xb2-50e_custom_boundary_v2/'
    'best_coco_segm_mAP_iter_19600.pth')

E1B_MODELS = [
    {
        'name': 'boundary_v2_default',
        'label': 'V2 default',
        'config':
        'configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v2.py',
        'checkpoint': V2_CHECKPOINT,
    },
    {
        'name': 'boundary_v2_masknms075_score03',
        'label': 'maskNMS0.75 score0.3',
        'config':
        'configs/ai4boundary/'
        'mask2former_r50_1xb2-50e_custom_boundary_v2_e1_masknms075.py',
        'checkpoint': V2_CHECKPOINT,
    },
    {
        'name': 'boundary_v2_masknms085_score02',
        'label': 'maskNMS0.85 score0.2',
        'config':
        'configs/ai4boundary/'
        'mask2former_r50_1xb2-50e_custom_boundary_v2_e1b_masknms085_score02.py',
        'checkpoint': V2_CHECKPOINT,
    },
    {
        'name': 'boundary_v2_masknms09_score02',
        'label': 'maskNMS0.90 score0.2',
        'config':
        'configs/ai4boundary/'
        'mask2former_r50_1xb2-50e_custom_boundary_v2_e1b_masknms09_score02.py',
        'checkpoint': V2_CHECKPOINT,
    },
    {
        'name': 'boundary_v2_argmax_strict',
        'label': 'argmax strict',
        'config':
        'configs/ai4boundary/'
        'mask2former_r50_1xb2-50e_custom_boundary_v2_e1_argmax.py',
        'checkpoint': V2_CHECKPOINT,
    },
    {
        'name': 'boundary_v2_argmax_balanced',
        'label': 'argmax balanced',
        'config':
        'configs/ai4boundary/'
        'mask2former_r50_1xb2-50e_custom_boundary_v2_e1b_argmax_balanced.py',
        'checkpoint': V2_CHECKPOINT,
    },
    {
        'name': 'boundary_v2_argmax_recall',
        'label': 'argmax recall',
        'config':
        'configs/ai4boundary/'
        'mask2former_r50_1xb2-50e_custom_boundary_v2_e1b_argmax_recall.py',
        'checkpoint': V2_CHECKPOINT,
    },
]


if __name__ == '__main__':
    eval_compare.MODELS = E1B_MODELS
    eval_compare.main()
