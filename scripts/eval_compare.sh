#!/bin/bash
# 单机 H800 直接跑多 checkpoint 对比评估。
set -euo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

python eval_compare.py --device cuda:0 "$@"
