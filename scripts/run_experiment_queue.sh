#!/bin/bash
# ============================================================
# 单卡 H800 顺序跑完 backbone 对照 + 三大问题消融矩阵。
#
# 单卡机器没法并行跑多个完整训练，这里全部串行执行（用 && 链接，前一个
# 失败就停下，不会在坏掉的环境里继续烧卡时）。当前 v4_combined 训练还在
# 跑，不要在它结束前启动本脚本。
#
# 大致耗时预估（基于 v4_combined 实测 ~1.5s/iter，H800 单卡）：
#   Phase A0 三次显存探测（各 2 iter）                    ≈ 数分钟
#   Phase A 四个 backbone（各 ~20000 iters，~8h/个）      ≈ 32h
#     (R101 若触发梯度累积回退，iters 翻倍但单步计算量减半，总耗时不变)
#   Phase B 五个消融实验（各 16500 iters，~7h/个）        ≈ 35h
#   合计 ≈ 67h（不含已经在跑的 v4_combined ~7h）
#
# 如果时间预算紧张，把下面 SCREEN_ITERS 从 0（=不截断，跑满）改成一个
# 更小的值（比如 10000），可以先用缩短预算筛选 backbone 排名，选出赢家
# 后再单独用完整预算重训一次；消融实验的 5 个不建议缩短，因为它们要和
# 已经用完整预算训练的 v4_combined 直接比。注意 SCREEN_ITERS 不会自动
# 适配 R101 梯度累积版的 2x iters，如果触发了回退请手动调整。
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

SCREEN_ITERS="${SCREEN_ITERS:-0}"   # 0 = 不截断，用配置自带的 max_iters
EXTRA_OPTS=()
if [ "${SCREEN_ITERS}" != "0" ]; then
  EXTRA_OPTS=(--cfg-options "train_cfg.max_iters=${SCREEN_ITERS}")
  echo ">>> SCREEN_ITERS=${SCREEN_ITERS}，backbone 对照阶段使用缩短预算"
fi

echo "================= Phase A0: 显存可行性探测（2 iter 干跑，不做 eval）================="
bash scripts/probe_batch_size.sh mask2former_r101_1xb2-50e_custom_boundary_v2
R101_CFG="mask2former_r101_1xb2-50e_custom_boundary_v2"
if grep -qi "out of memory" "/tmp/probe_mask2former_r101_1xb2-50e_custom_boundary_v2.log" 2>/dev/null; then
  echo ">>> R101 在 batch_size=12 下 OOM，自动切换到梯度累积版（等效 batch 不变）"
  R101_CFG="mask2former_r101_1xb2-50e_custom_boundary_v2_gradaccum"
fi
bash scripts/probe_batch_size.sh mask2former_swin-t_1xb2-50e_custom_boundary_v2
bash scripts/probe_batch_size.sh mask2former_hrnetv2p-w18_1xb2-50e_custom_boundary_v2

echo "================= Phase A: backbone 对照（固定 V2 loss/数据配方）================="

echo ">>> [A1/4] R50（V2 本身，重新训一遍以保证同一台机器/同一份代码可比）"
python tools/train.py configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v2.py "${EXTRA_OPTS[@]}"

echo ">>> [A2/4] R101"
python tools/train.py "configs/ai4boundary/${R101_CFG}.py" "${EXTRA_OPTS[@]}"

echo ">>> [A3/4] Swin-T"
python tools/train.py configs/ai4boundary/mask2former_swin-t_1xb2-50e_custom_boundary_v2.py "${EXTRA_OPTS[@]}"

echo ">>> [A4/4] HRNetV2p-W18"
python tools/train.py configs/ai4boundary/mask2former_hrnetv2p-w18_1xb2-50e_custom_boundary_v2.py "${EXTRA_OPTS[@]}"

echo "================= Phase B: 三大问题消融矩阵（在 v3_clean_bg 数据配方上）================="

echo ">>> [B1/5] 仅 Track2 surface loss"
python tools/train.py configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v4b_surface_only.py

echo ">>> [B2/5] 仅 Track3 curvature loss"
python tools/train.py configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v4b_curvature_only.py

echo ">>> [B3/5] v4_combined 去掉 surface loss"
python tools/train.py configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v4_minus_surface.py

echo ">>> [B4/5] v4_combined 去掉 curvature loss"
python tools/train.py configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v4_minus_curvature.py

echo ">>> [B5/5] v4_combined 去掉 Track1 后处理"
python tools/train.py configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v4_minus_track1.py

echo "================= 全部实验完成 ================="
