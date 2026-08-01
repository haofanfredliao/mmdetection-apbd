#!/bin/bash
# 单机 H800 冒烟测试：只跑 20 个 iter，确认配置能跑通。
set -euo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

python tools/train.py \
  configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v4_combined.py \
  --work-dir work_dirs/smoke_mask2former_boundary_v4_combined \
  --cfg-options \
    train_dataloader.batch_size=2 \
    train_dataloader.num_workers=4 \
    train_cfg.max_iters=20 \
    train_cfg.val_interval=1000 \
    default_hooks.checkpoint.interval=1000 \
    custom_hooks.0.end_iter=10
