_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v2.py']

# E2: train-time non-overlap prior.
# Matched positive queries correspond one-to-one to non-overlapping field GT
# instances, so this loss discourages the model from assigning the same pixels
# to multiple positive masks.
model = dict(
    panoptic_head=dict(
        loss_nonoverlap=dict(
            type='NonOverlapLoss',
            loss_weight=1.0,
            mode='sum_excess',
            power=2.0),
        nonoverlap_max_res=128))
