#!/bin/bash
# ============================================================
# 单卡 H800 顺序跑完 3 大问题的消融矩阵（5 个实验），从
# run_experiment_queue.sh 的 Phase B 拆出来单独跑，因为 backbone
# 对照实验（Phase A）先不做。
#
# 5 个实验都在 v3_clean_bg 数据配方上做加/减法：
#   B1 仅加 Track2 surface loss（Kervadec）
#   B2 仅加 Track3 curvature loss（正则化）
#   B3 v4_combined 去掉 surface loss
#   B4 v4_combined 去掉 curvature loss
#   B5 v4_combined 去掉 Track1 后处理（argmax/mask_nms）
# 加上已经训练完的 v3_clean_bg（全部消融基线）和 v4_combined（全部叠加），
# 一共能拼出完整的消融对照表。
#
# 每个实验 max_iters=16500，实测 v4_combined ~1.5s/iter，预计每个
# 实验 ~7h，5 个合计 ≈ 35h。用 && 串行链接，前一个失败就停下。
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

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

echo "================= 消融实验全部完成 ================="
