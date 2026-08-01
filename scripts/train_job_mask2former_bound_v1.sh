#!/bin/bash
# 单机 H800 直接训练 Mask2Former + Boundary Dice Loss V1。
set -euo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

python tools/train.py configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v1.py "$@"
