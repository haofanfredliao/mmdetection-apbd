import numpy as np
from typing import Sequence, Dict
from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS

@METRICS.register_module()
class FieldSegmentationMetric(BaseMetric):
    """
    定制的农田分割评估指标，计算过分割(OS)和欠分割(US)率。
    """
    def __init__(self, iou_thr=0.5, **kwargs):
        super().__init__(**kwargs)
        self.iou_thr = iou_thr

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """处理每个 batch 的预测结果和真实标签"""
        for data_sample in data_samples:
            pred_masks = data_sample['pred_instances']['masks'].cpu().numpy()
            gt_masks = data_sample['gt_instances']['masks'].cpu().numpy()
            
            # 记录单张图像的统计信息
            self.results.append({
                'pred_masks': pred_masks,
                'gt_masks': gt_masks
            })

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """计算整个验证集的 OS 和 US"""
        total_gt = 0
        total_pred = 0
        over_segmented_count = 0
        under_segmented_count = 0

        for res in results:
            pred_masks = res['pred_masks'] # [M, H, W]
            gt_masks = res['gt_masks']     # [N, H, W]
            
            total_pred += len(pred_masks)
            total_gt += len(gt_masks)
            
            if len(pred_masks) == 0 or len(gt_masks) == 0:
                continue

            # 计算交集矩阵 [N, M]
            intersection = np.zeros((len(gt_masks), len(pred_masks)))
            for i, gt in enumerate(gt_masks):
                for j, pred in enumerate(pred_masks):
                    intersection[i, j] = np.logical_and(gt, pred).sum()

            # 计算 GT 面积和 Pred 面积
            gt_areas = gt_masks.sum(axis=(1, 2))
            pred_areas = pred_masks.sum(axis=(1, 2))

            # --- 欠分割 (Under-segmentation) 统计 ---
            # 如果一个 Pred 覆盖了多个 GT（交集占 GT 面积比例 > thr）
            for j in range(len(pred_masks)):
                # 找到与该 pred 有显著交集的 GT 数量
                covered_gts = (intersection[:, j] / (gt_areas + 1e-5)) > self.iou_thr
                if covered_gts.sum() > 1:
                    under_segmented_count += 1

            # --- 过分割 (Over-segmentation) 统计 ---
            # 如果一个 GT 被多个 Pred 覆盖（交集占 Pred 面积比例 > thr）
            for i in range(len(gt_masks)):
                # 找到覆盖该 GT 的 pred 数量
                covering_preds = (intersection[i, :] / (pred_areas + 1e-5)) > self.iou_thr
                if covering_preds.sum() > 1:
                    over_segmented_count += 1

        # 计算比例
        os_rate = over_segmented_count / max(total_gt, 1)
        us_rate = under_segmented_count / max(total_pred, 1)

        return {
            'Over-segmentation_Rate': os_rate,
            'Under-segmentation_Rate': us_rate
        }