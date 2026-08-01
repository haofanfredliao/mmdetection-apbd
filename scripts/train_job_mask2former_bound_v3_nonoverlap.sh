#!/bin/bash
# 单机 H800 直接训练 Mask2Former + 非重叠先验 Loss V3（原 shard 配置，
# 单卡满显存跑，不再需要 shard 版本的梯度累积设置）。
set -euo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

python tools/train.py configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v3_nonoverlap.py "$@"
