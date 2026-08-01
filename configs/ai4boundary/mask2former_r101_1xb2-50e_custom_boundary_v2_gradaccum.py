# ============================================================
# ResNet101 备用方案 — 梯度累积版
#
# 如果 mask2former_r101_1xb2-50e_custom_boundary_v2.py 在 batch_size=12
# 下探测到 OOM（用 scripts/probe_batch_size.sh 验证），用这个配置代替：
# micro-batch 降到 6，累积 2 步再更新参数，等效 batch 仍是 12，显存峰值
# 大约减半。为了让"见过的数据量/优化器更新次数"和其它 backbone 保持一致
# （对比实验的公平性），max_iters/val_interval/checkpoint interval/学习率
# 里程碑全部按 2x 等比放大，不是简单减半 batch 就完事。
#
# 代价：单个 iteration 里做 2 次 forward+backward 再更新一次参数，总
# forward/backward 计算量和直接用 batch=12 一样，所以总耗时基本不变，只是
# 显存换成了时间没有的额外开销（可忽略）。
# ============================================================

_base_ = ['./mask2former_r101_1xb2-50e_custom_boundary_v2.py']

train_dataloader = dict(batch_size=6)

optim_wrapper = dict(accumulative_counts=2)

max_iters = 40000
val_interval_iters = 1600

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=val_interval_iters)

param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[int(max_iters * 0.9), int(max_iters * 0.95)],
    gamma=0.1)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=val_interval_iters,
        max_keep_ckpts=3,
        save_best='coco/segm_mAP',
        rule='greater'))

work_dir = './work_dirs/mask2former_r101_1xb2-50e_custom_boundary_v2_gradaccum'
