# ============================================================
# Backbone comparison — Swin-T on the V5 recipe
#
# Holds the entire V5 recipe fixed (BoundaryDiceLoss + NonOverlapLoss +
# point-sampled KervadecBoundaryLoss, both ramps, argmax fusion head) and
# swaps only the backbone. paramwise_cfg follows the official
# mask2former_swin-t panoptic config (backbone lr_mult=0.1, norm / pos-embed
# / relative-bias decay_mult=0.0).
#
# Schedule matches R50-V5 exactly (16500 iters, val every 660) so the
# backbone comparison is not confounded by training length. The older
# Swin-T V2 config used 20000 iters; do not reinherit that.
# ============================================================

_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v5.py']

pretrained = './pretrain/swin_tiny_patch4_window7_224.pth'
depths = [2, 2, 6, 2]

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    panoptic_head=dict(in_channels=[96, 192, 384, 768]))

# ---------------- optimizer: Swin-specific paramwise_cfg ----------------
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi,
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})

optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))

work_dir = './work_dirs/mask2former_swin-t_1xb2-50e_custom_boundary_v5'
