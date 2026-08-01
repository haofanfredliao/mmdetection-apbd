#!/bin/bash
# ============================================================
# 显存可行性探测：用当前配置里的 batch_size 跑 2 个 iter（不做 eval），
# 看会不会 OOM。跑完立刻清理产物，不影响正式训练。
#
# 用法: scripts/probe_batch_size.sh <config_name_without_.py> [batch_size]
# 例如: scripts/probe_batch_size.sh mask2former_r101_1xb2-50e_custom_boundary_v2
#       scripts/probe_batch_size.sh mask2former_r101_1xb2-50e_custom_boundary_v2 8
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

CFG_NAME="$1"
BATCH="${2:-}"
CFG_PATH="configs/ai4boundary/${CFG_NAME}.py"
PROBE_DIR="work_dirs/_probe_${CFG_NAME}"
LOG_FILE="/tmp/probe_${CFG_NAME}.log"

EXTRA_OPTS=(--cfg-options
  train_cfg.max_iters=2
  train_cfg.val_interval=100000
  default_hooks.checkpoint.interval=100000
  default_hooks.logger.interval=1)
if [ -n "${BATCH}" ]; then
  EXTRA_OPTS+=(train_dataloader.batch_size="${BATCH}")
fi

echo ">>> probing ${CFG_PATH} (batch_size override: ${BATCH:-<config default>}) ..."
timeout 300 python tools/train.py "${CFG_PATH}" \
  --work-dir "${PROBE_DIR}" \
  "${EXTRA_OPTS[@]}" \
  > "${LOG_FILE}" 2>&1
STATUS=$?

PEAK_MEM=$(grep -o "memory: [0-9]*" "${LOG_FILE}" | tail -1 | awk '{print $2}')

if grep -qi "out of memory" "${LOG_FILE}"; then
  echo "!!! OOM: ${CFG_PATH} 在这个 batch_size 下会爆显存，日志见 ${LOG_FILE}"
elif [ "${STATUS}" -eq 0 ] && [ -n "${PEAK_MEM}" ]; then
  echo ">>> OK: ${CFG_PATH} 跑通，峰值显存约 ${PEAK_MEM} MiB（详见 ${LOG_FILE}）"
else
  echo "??? 探测异常退出（code=${STATUS}），非 OOM 相关错误，请查看 ${LOG_FILE}"
fi

rm -rf "${PROBE_DIR}"
