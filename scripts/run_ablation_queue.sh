#!/bin/bash
# ============================================================
# 消融矩阵（Track3 废弃后的坍缩版）
#
# 原计划是 5 个完整训练（约 35 GPU-小时）。删掉 Track3 之后只剩两个因子，
# 而其中 Track1 是纯推理期改动（只换 panoptic_fusion_head + test_cfg，
# 训练图完全不变），所以整个 2x2 矩阵只需要 1 次训练：
#
#                  | 无 Track1              | 有 Track1
#   ---------------+------------------------+---------------------------
#   无 surface     | v3_clean_bg（已训完）  | v3 权重 + 换 test_cfg
#   有 surface     | 本脚本训练的这一次     | 同一份权重 + 换 test_cfg
#
# 也就是 1 次训练（~7h）+ 3 次纯推理评估，从 35 GPU-小时降到 ~7。
#
# 已作废、不要再跑的配置（保留文件仅作记录）：
#   v4b_curvature_only  —— Track3 已废弃
#   v4_minus_curvature  —— 删掉 curvature 后与 v4_combined 完全相同
#   v4_minus_track1     —— 与 v4b_surface_only 训练图相同
#   v4_minus_surface    —— 等价于 v3 权重 + Track1 后处理，纯推理可得
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

TRAIN_CFG=configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v4b_surface_only.py
WORK_DIR=work_dirs/mask2former_r50_1xb2-50e_custom_boundary_v4b_surface_only

echo "================= [1/2] 训练：v3 配方 + Track2 surface loss（修正权重后）================="
python tools/train.py "${TRAIN_CFG}"

echo "================= [2/2] 后处理评估：同一份权重叠加 Track1 ================="
CKPT=$(ls -t "${WORK_DIR}"/best_coco_segm_mAP_iter_*.pth | head -1)
echo ">>> 使用 checkpoint: ${CKPT}"

# 无 Track1 的那一格训练时已经评过，这里补上 +Track1 的一格。
python tools/test.py "${TRAIN_CFG}" "${CKPT}" \
  --work-dir outputs/eval/surface_plus_track1 \
  --cfg-options model.panoptic_fusion_head.type=FieldMaskFormerFusionHead \
                model.test_cfg.argmax_instance=True \
                model.test_cfg.filter_low_score=True \
                model.test_cfg.score_thr=0.2 \
                model.test_cfg.iou_thr=0.7

echo "================= 完成，四格结果分别在 ================="
echo "  无surface 无Track1: work_dirs/..._v3_clean_bg（训练日志）/ outputs/eval/ 下的 v3 评估"
echo "  无surface 有Track1: outputs/eval/v3_plus_track1"
echo "  有surface 无Track1: ${WORK_DIR}（训练日志末次 eval）"
echo "  有surface 有Track1: outputs/eval/surface_plus_track1"
