#!/bin/bash
# Mask R-CNN baseline aligned with V5 data/eval controls.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
exec /root/miniconda3/envs/openmmlab/bin/python tools/train.py \
  configs/ai4boundary/mask-rcnn_r50_fpn_50e_ai4b_v5data.py \
  "$@"
