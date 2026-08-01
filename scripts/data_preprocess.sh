#!/bin/bash
# 单机 H800 直接跑数据预处理（COCO 格式转换）。
set -euo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

python convert_to_coco.py "$@"
