#!/bin/bash
# 单机 H800 直接训练 Mask R-CNN baseline。
set -euo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

python tools/train.py configs/ai4boundary/mask-rcnn_r50_fpn_1x_ai4b.py "$@"
