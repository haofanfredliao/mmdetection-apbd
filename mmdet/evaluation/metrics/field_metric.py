from typing import Dict, Sequence

import cv2
import numpy as np
from mmengine.evaluator import BaseMetric

from mmdet.registry import METRICS


@METRICS.register_module()
class FieldSegmentationMetric(BaseMetric):
    """
    定制的农田分割评估指标。

    除过/欠分割率外，额外输出边界密合、边界规整和实例计数指标，
    用于窄类别农田实例分割的统一实验对比。
    """

    def __init__(self,
                 iou_thr=0.5,
                 match_iou_thr=0.5,
                 duplicate_iou_thr=0.75,
                 boundary_iou_radius=3,
                 boundary_f_tolerances=(1, 3, 5),
                 regularity_iou_thr=0.95,
                 regularity_eps=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0,
                                 5.0, 7.0, 10.0, 15.0, 20.0),
                 regularity_max_instances=2000,
                 **kwargs):
        super().__init__(**kwargs)
        self.iou_thr = iou_thr
        self.match_iou_thr = match_iou_thr
        self.duplicate_iou_thr = duplicate_iou_thr
        self.boundary_iou_radius = boundary_iou_radius
        self.boundary_f_tolerances = tuple(boundary_f_tolerances)
        self.regularity_iou_thr = regularity_iou_thr
        self.regularity_eps = tuple(regularity_eps)
        self.regularity_max_instances = regularity_max_instances

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """处理每个 batch 的预测结果和真实标签"""
        for data_sample in data_samples:
            pred_masks = data_sample['pred_instances']['masks']
            pred_masks = self._to_numpy_masks(pred_masks)

            pred_scores = data_sample['pred_instances'].get('scores', None)
            if pred_scores is not None and hasattr(pred_scores, 'cpu'):
                pred_scores = pred_scores.cpu().numpy()

            gt_masks = data_sample['gt_instances']['masks']
            gt_masks = self._to_numpy_masks(gt_masks)

            # 记录单张图像的统计信息
            self.results.append({
                'pred_masks': pred_masks,
                'gt_masks': gt_masks,
                'pred_scores': pred_scores,
            })

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """计算整个验证集的统一农田分割指标"""
        total_gt = 0
        total_pred = 0
        over_segmented_count = 0
        under_segmented_count = 0
        duplicate_pred_count = 0

        matched_boundary_ious = []
        matched_boundary_f = {
            tol: []
            for tol in self.boundary_f_tolerances
        }
        pred_vertices = []
        pred_curvature_energy = []

        for res in results:
            pred_masks = res['pred_masks'].astype(bool)  # [M, H, W]
            gt_masks = res['gt_masks'].astype(bool)      # [N, H, W]

            total_pred += len(pred_masks)
            total_gt += len(gt_masks)

            duplicate_pred_count += self._count_duplicate_predictions(
                pred_masks, res.get('pred_scores'))

            for pred in pred_masks:
                if len(pred_vertices) < self.regularity_max_instances:
                    vertices = self._vertices_at_iou(
                        pred, self.regularity_iou_thr)
                    if np.isfinite(vertices):
                        pred_vertices.append(vertices)

                curvature = self._curvature_energy(pred)
                if np.isfinite(curvature):
                    pred_curvature_energy.append(curvature)

            if len(pred_masks) == 0 or len(gt_masks) == 0:
                continue

            intersection, union, gt_areas, pred_areas = self._mask_overlap(
                gt_masks, pred_masks)
            ious = intersection / (union + 1e-5)

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

            for gt_idx, pred_idx in self._greedy_match(ious):
                if ious[gt_idx, pred_idx] < self.match_iou_thr:
                    continue
                gt = gt_masks[gt_idx]
                pred = pred_masks[pred_idx]
                matched_boundary_ious.append(
                    self._boundary_iou(pred, gt, self.boundary_iou_radius))
                for tol in self.boundary_f_tolerances:
                    matched_boundary_f[tol].append(
                        self._boundary_f_score(pred, gt, tol))

        # 计算比例
        os_rate = over_segmented_count / max(total_gt, 1)
        us_rate = under_segmented_count / max(total_pred, 1)

        metrics = {
            'Over-segmentation_Rate': os_rate,
            'Under-segmentation_Rate': us_rate,
            'Pred_GT_Count_Ratio': total_pred / max(total_gt, 1),
            'Duplicate_Rate': duplicate_pred_count / max(total_pred, 1),
            'Boundary-IoU': self._nanmean(matched_boundary_ious),
            'Vertices_IoU95_pred': self._nanmean(pred_vertices),
            'Curvature_Energy_pred': self._nanmean(pred_curvature_energy),
        }

        for tol in self.boundary_f_tolerances:
            metrics[f'Boundary-F_{tol}px'] = self._nanmean(
                matched_boundary_f[tol])

        return metrics

    @staticmethod
    def _to_numpy_masks(masks):
        if hasattr(masks, 'cpu'):
            masks = masks.cpu().numpy()
        elif hasattr(masks, 'to_ndarray'):
            masks = masks.to_ndarray()
        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[None, ...]
        return masks.astype(bool)

    @staticmethod
    def _mask_overlap(gt_masks, pred_masks):
        intersection = np.zeros((len(gt_masks), len(pred_masks)),
                                dtype=np.float64)
        for i, gt in enumerate(gt_masks):
            for j, pred in enumerate(pred_masks):
                intersection[i, j] = np.logical_and(gt, pred).sum()

        gt_areas = gt_masks.sum(axis=(1, 2)).astype(np.float64)
        pred_areas = pred_masks.sum(axis=(1, 2)).astype(np.float64)
        union = gt_areas[:, None] + pred_areas[None, :] - intersection
        return intersection, union, gt_areas, pred_areas

    @staticmethod
    def _greedy_match(ious):
        if ious.size == 0:
            return []
        candidates = np.argwhere(ious > 0)
        order = np.argsort(ious[candidates[:, 0], candidates[:, 1]])[::-1]
        matched_gt = set()
        matched_pred = set()
        matches = []
        for idx in order:
            gt_idx, pred_idx = candidates[idx]
            gt_idx = int(gt_idx)
            pred_idx = int(pred_idx)
            if gt_idx in matched_gt or pred_idx in matched_pred:
                continue
            matches.append((gt_idx, pred_idx))
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
        return matches

    def _count_duplicate_predictions(self, pred_masks, pred_scores=None):
        if len(pred_masks) <= 1:
            return 0

        if pred_scores is None:
            order = np.arange(len(pred_masks))
        else:
            order = np.argsort(np.asarray(pred_scores))[::-1]

        kept = []
        duplicate_count = 0
        for idx in order:
            pred = pred_masks[idx]
            is_duplicate = False
            for kept_idx in kept:
                iou = self._binary_iou(pred, pred_masks[kept_idx])
                if iou >= self.duplicate_iou_thr:
                    is_duplicate = True
                    break
            if is_duplicate:
                duplicate_count += 1
            else:
                kept.append(idx)
        return duplicate_count

    @staticmethod
    def _binary_iou(mask_a, mask_b):
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        return float(inter / (union + 1e-5))

    @staticmethod
    def _boundary(mask):
        mask_u8 = mask.astype(np.uint8)
        if mask_u8.sum() == 0:
            return np.zeros_like(mask_u8, dtype=bool)
        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded = cv2.erode(mask_u8, kernel, iterations=1)
        return (mask_u8 > 0) & (eroded == 0)

    def _boundary_band(self, mask, radius):
        boundary = self._boundary(mask).astype(np.uint8)
        if boundary.sum() == 0:
            return boundary.astype(bool)
        if radius <= 0:
            return boundary.astype(bool)
        kernel_size = 2 * int(radius) + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        band = cv2.dilate(boundary, kernel, iterations=1)
        return band.astype(bool)

    def _boundary_iou(self, pred, gt, radius):
        pred_band = self._boundary_band(pred, radius)
        gt_band = self._boundary_band(gt, radius)
        return self._binary_iou(pred_band, gt_band)

    def _boundary_f_score(self, pred, gt, tolerance):
        pred_boundary = self._boundary(pred)
        gt_boundary = self._boundary(gt)
        pred_count = pred_boundary.sum()
        gt_count = gt_boundary.sum()
        if pred_count == 0 and gt_count == 0:
            return 1.0
        if pred_count == 0 or gt_count == 0:
            return 0.0

        gt_dist = cv2.distanceTransform((~gt_boundary).astype(np.uint8),
                                        cv2.DIST_L2, 3)
        pred_dist = cv2.distanceTransform((~pred_boundary).astype(np.uint8),
                                          cv2.DIST_L2, 3)
        precision = (gt_dist[pred_boundary] <= tolerance).mean()
        recall = (pred_dist[gt_boundary] <= tolerance).mean()
        return float(2 * precision * recall / (precision + recall + 1e-5))

    def _vertices_at_iou(self, mask, target_iou):
        contours = self._external_contours(mask)
        if not contours:
            return np.nan

        best_vertices = np.inf
        for eps in self.regularity_eps:
            simplified = np.zeros(mask.shape, dtype=np.uint8)
            vertices = 0
            for contour in contours:
                approx = cv2.approxPolyDP(contour, epsilon=float(eps), closed=True)
                if len(approx) < 3:
                    approx = contour
                vertices += len(approx)
                cv2.drawContours(simplified, [approx], -1, 1, thickness=-1)

            if self._binary_iou(mask, simplified.astype(bool)) >= target_iou:
                best_vertices = min(best_vertices, vertices)

        return float(best_vertices) if np.isfinite(best_vertices) else np.nan

    def _curvature_energy(self, mask):
        contours = self._external_contours(mask)
        if not contours:
            return np.nan

        energies = []
        for contour in contours:
            points = contour.reshape(-1, 2).astype(np.float32)
            if len(points) < 3:
                continue
            prev_points = np.roll(points, 1, axis=0)
            next_points = np.roll(points, -1, axis=0)
            v1 = points - prev_points
            v2 = next_points - points
            norm1 = np.linalg.norm(v1, axis=1)
            norm2 = np.linalg.norm(v2, axis=1)
            valid = (norm1 > 1e-5) & (norm2 > 1e-5)
            if not valid.any():
                continue
            v1 = v1[valid] / norm1[valid, None]
            v2 = v2[valid] / norm2[valid, None]
            cos_angle = np.clip((v1 * v2).sum(axis=1), -1.0, 1.0)
            angles = np.arccos(cos_angle)
            energies.append(float(np.mean(angles**2)))

        return self._nanmean(energies)

    @staticmethod
    def _external_contours(mask):
        mask_u8 = mask.astype(np.uint8)
        if mask_u8.sum() == 0:
            return []
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        return [cnt for cnt in contours if len(cnt) >= 3]

    @staticmethod
    def _nanmean(values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return 0.0
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return 0.0
        return float(finite_values.mean())