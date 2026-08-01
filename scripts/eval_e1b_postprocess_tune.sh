#!/bin/bash
# 单机 H800 直接跑 E1b 后处理精调评估。
set -euo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

python eval_e1b_postprocess_tune.py --device cuda:0 "$@"
