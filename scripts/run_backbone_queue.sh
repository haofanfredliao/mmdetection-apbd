#!/usr/bin/env bash
# Sequential backbone comparison on the V5 recipe.
# Order: Swin-T (most informative) → HRNet-W18 → R101 (capacity-only).
# Each run uses the same schedule / effective batch as R50-V5.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

PY=/root/miniconda3/envs/openmmlab/bin/python
CONFIGS=(
  configs/ai4boundary/mask2former_swin-t_1xb2-50e_custom_boundary_v5.py
  configs/ai4boundary/mask2former_hrnetv2p-w18_1xb2-50e_custom_boundary_v5.py
  configs/ai4boundary/mask2former_r101_1xb2-50e_custom_boundary_v5.py
)
NAMES=(swin_t hrnet_w18 r101)

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  name="${NAMES[$i]}"
  log="logs/backbone_${name}_v5.log"
  echo "======== $(date '+%F %T')  START ${name}  ========" | tee -a logs/backbone_queue.log
  echo "config: ${cfg}" | tee -a logs/backbone_queue.log
  # shellcheck disable=SC2086
  ${PY} tools/train.py "${cfg}" 2>&1 | tee "${log}"
  echo "======== $(date '+%F %T')  DONE  ${name}  ========" | tee -a logs/backbone_queue.log
done

echo "All backbone runs finished at $(date '+%F %T')" | tee -a logs/backbone_queue.log
