# ============================================================
# Backbone 对照实验 — Swin-T
#
# 固定 V2 的 loss/数据配方（BoundaryDiceLoss + QualityAwareCocoDataset），
# 只替换 backbone，用于和 R50（V2 本身）、HRNetV2p-W18 做公平对比。
# paramwise_cfg 沿用官方 mask2former_swin-t...-panoptic.py 的设置（backbone
# lr_mult=0.1，norm/位置编码/相对位置偏置 decay_mult=0.0）。
# ============================================================

_base_ = ['./mask2former_r50_1xb2-50e_custom_boundary_v2.py']

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
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
# Keep the same base LR as V2 (0.00005); only add the differential lr/decay
# multipliers Swin needs (backbone lr_mult=0.1, norm/pos-embed decay=0.0).
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

# ---------------- fix inherited checkpoint-interval bug ----------------
# V2 sets checkpoint interval=5 assuming by_epoch semantics, but the loop
# is iteration-based (by_epoch=False) -> was saving every 5 ITERATIONS.
# Align checkpoint with eval, both every ~2 epochs (~800 iters here).
val_interval_iters = 800

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=20000,
    val_interval=val_interval_iters)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=val_interval_iters,
        max_keep_ckpts=3,
        save_best='coco/segm_mAP',
        rule='greater'))

work_dir = './work_dirs/mask2former_swin-t_1xb2-50e_custom_boundary_v2'

# NOTE: if this OOMs on a single H800 at batch_size=12 (inherited from V2),
# lower train_dataloader.batch_size (e.g. to 8) via --cfg-options rather
# than editing this file, so the change is visible in the run log.
