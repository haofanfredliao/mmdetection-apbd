#!/bin/bash
# 单机 H800 直接跑 E1 推理后处理参数扫描评估。
set -euo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

python eval_e1_inference_sweep.py --device cuda:0 "$@"
