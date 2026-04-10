# Copyright (c) OpenMMLab. All rights reserved.
# V2: adds boundary aux loss on full-resolution decoder outputs and
#     per-image quality loss weighting via img_meta['loss_weight'].
from typing import List, Tuple

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.utils import digit_version
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import reduce_mean
from ..utils import multi_apply
from .mask2former_head import Mask2FormerHead


@MODELS.register_module()
class Mask2FormerHeadV2(Mask2FormerHead):
    """Extends Mask2FormerHead with two improvements for agricultural field
    segmentation:

    1. **Boundary aux loss**: ``loss_boundary`` is computed on full-resolution
       (upsampled) decoder mask outputs, bypassing the point-sampling path so
       that morphological boundary extraction is spatially valid.

    2. **Per-image quality weighting**: if ``img_meta`` contains a
       ``loss_weight`` key (float), each GT instance from that image is scaled
       by that factor when computing ``loss_dice`` and ``loss_boundary``.
       ``3_extreme`` images are excluded at the dataset level;
       ``2_lazy`` images carry ``loss_weight=0.2``.

    Only ``loss_boundary`` and ``loss_dice`` are quality-weighted; the
    cross-entropy ``loss_mask`` is left unscaled for training stability.

    New config args (beyond Mask2FormerHead):
        loss_boundary (dict, optional): Config for the boundary aux loss.
            If None the boundary loss term is omitted.  Recommended:
            ``dict(type='BoundaryDiceLoss', loss_weight=2.0, kernel_size=3,
            eps=1e-5)``
    """

    def __init__(self, loss_boundary=None, boundary_max_res: int = 256,
                 **kwargs):
        super().__init__(**kwargs)
        self.loss_boundary = (
            MODELS.build(loss_boundary) if loss_boundary is not None else None)
        self.boundary_max_res = boundary_max_res

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_per_gt_quality_weights(self, mask_targets_list, batch_img_metas,
                                       reference_tensor):
        """Build a float tensor of shape (num_total_gts,) with per-gt loss
        weights derived from img_meta['loss_weight']."""
        chunks = []
        for i, t in enumerate(mask_targets_list):
            n_pos = t.shape[0]
            if n_pos == 0:
                continue
            w = float(batch_img_metas[i].get('loss_weight', 1.0))
            chunks.append(reference_tensor.new_full((n_pos,), w))
        if not chunks:
            return None
        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------
    # Override _loss_by_feat_single
    # ------------------------------------------------------------------

    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor,
                             batch_gt_instances, batch_img_metas,
                             compute_boundary: bool = False):
        """Loss for a single decoder layer.

        Adds quality weighting via img_meta['loss_weight'].
        Optionally computes boundary aux loss at capped resolution
        (only called with compute_boundary=True for the last decoder layer).
        """
        from mmcv.ops import point_sample
        from ..utils import get_uncertain_point_coords_with_randomness

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        (labels_list, label_weights_list, mask_targets_list,
         mask_weights_list, avg_factor) = self.get_targets(
             cls_scores_list, mask_preds_list,
             batch_gt_instances, batch_img_metas)

        labels = torch.stack(labels_list, dim=0)
        label_weights = torch.stack(label_weights_list, dim=0)
        mask_targets = torch.cat(mask_targets_list, dim=0)   # (N_gt, H_gt, W_gt)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # --- classification loss ---
        cls_scores_flat = cls_scores.flatten(0, 1)
        labels_flat = labels.flatten(0, 1)
        label_weights_flat = label_weights.flatten(0, 1)
        class_weight = cls_scores_flat.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores_flat, labels_flat, label_weights_flat,
            avg_factor=class_weight[labels_flat].sum())

        num_total_masks = reduce_mean(cls_scores_flat.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # --- extract positive mask predictions ---
        # mask_preds shape: (B, Q, h_feat, w_feat)  e.g. (B, Q, 256, 256)
        mask_preds_pos = mask_preds[mask_weights > 0]  # (N_gt, h_feat, w_feat)

        if mask_targets.shape[0] == 0:
            zero = mask_preds_pos.sum()
            if compute_boundary and self.loss_boundary is not None:
                return loss_cls, zero, zero, zero
            return loss_cls, zero, zero

        # --- per-gt quality weights ---
        per_gt_w = self._build_per_gt_quality_weights(
            mask_targets_list, batch_img_metas, mask_preds_pos)

        # ------------------------------------------------------------------
        # Boundary aux loss – computed at decoder feature resolution,
        # capped at boundary_max_res.  GT mask is downsampled to match.
        # No upsampling of pred needed → much lower memory footprint.
        # Only runs when compute_boundary=True (last decoder layer only).
        # ------------------------------------------------------------------
        if compute_boundary and self.loss_boundary is not None:
            h_feat, w_feat = mask_preds_pos.shape[-2:]
            bdr_h = min(h_feat, self.boundary_max_res)
            bdr_w = min(w_feat, self.boundary_max_res)

            # pred: shrink only if feature resolution exceeds boundary_max_res
            if bdr_h < h_feat or bdr_w < w_feat:
                mask_preds_bdr = F.interpolate(
                    mask_preds_pos.unsqueeze(1), size=(bdr_h, bdr_w),
                    mode='bilinear', align_corners=False).squeeze(1)
            else:
                mask_preds_bdr = mask_preds_pos  # already at or below cap

            # GT: downsample from full image res to boundary res
            with torch.no_grad():
                mask_targets_bdr = F.interpolate(
                    mask_targets.unsqueeze(1).float(), size=(bdr_h, bdr_w),
                    mode='nearest').squeeze(1)

            loss_boundary = self.loss_boundary(
                mask_preds_bdr, mask_targets_bdr,
                weight=per_gt_w, avg_factor=num_total_masks)
        else:
            loss_boundary = None

        # ------------------------------------------------------------------
        # Standard point-sampled dice + CE mask losses (quality-weighted dice)
        # ------------------------------------------------------------------
        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds_pos.unsqueeze(1), None,
                self.num_points, self.oversample_ratio,
                self.importance_sample_ratio)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(),
                points_coords).squeeze(1)

        mask_point_preds = point_sample(
            mask_preds_pos.unsqueeze(1), points_coords).squeeze(1)

        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets,
            weight=per_gt_w, avg_factor=num_total_masks)

        mask_point_preds_flat = mask_point_preds.reshape(-1)
        mask_point_targets_flat = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds_flat, mask_point_targets_flat,
            avg_factor=num_total_masks * self.num_points)

        if loss_boundary is not None:
            return loss_cls, loss_mask, loss_dice, loss_boundary
        return loss_cls, loss_mask, loss_dice

    # ------------------------------------------------------------------
    # Override loss_by_feat
    # ------------------------------------------------------------------

    def loss_by_feat(self, all_cls_scores, all_mask_preds,
                     batch_gt_instances, batch_img_metas):
        """Compute losses across decoder layers.

        For all-but-last layers: standard cls/mask/dice losses only.
        For the last layer: additionally compute boundary aux loss
        (once, at capped resolution).
        """
        num_dec_layers = len(all_cls_scores)

        # Aux decoder layers (all but last) — no boundary loss
        if num_dec_layers > 1:
            aux_results = multi_apply(
                self._loss_by_feat_single,
                list(all_cls_scores[:-1]),
                list(all_mask_preds[:-1]),
                [batch_gt_instances] * (num_dec_layers - 1),
                [batch_img_metas] * (num_dec_layers - 1))
            aux_cls, aux_mask, aux_dice = aux_results
        else:
            aux_cls, aux_mask, aux_dice = [], [], []

        # Last decoder layer — with boundary loss
        last_result = self._loss_by_feat_single(
            all_cls_scores[-1], all_mask_preds[-1],
            batch_gt_instances, batch_img_metas,
            compute_boundary=(self.loss_boundary is not None))

        if self.loss_boundary is not None:
            last_cls, last_mask, last_dice, last_boundary = last_result
        else:
            last_cls, last_mask, last_dice = last_result

        loss_dict = dict()
        loss_dict['loss_cls'] = last_cls
        loss_dict['loss_mask'] = last_mask
        loss_dict['loss_dice'] = last_dice
        if self.loss_boundary is not None:
            loss_dict['loss_boundary'] = last_boundary

        for dec_i, (lc, lm, ld) in enumerate(
                zip(aux_cls, aux_mask, aux_dice)):
            loss_dict[f'd{dec_i}.loss_cls'] = lc
            loss_dict[f'd{dec_i}.loss_mask'] = lm
            loss_dict[f'd{dec_i}.loss_dice'] = ld

        return loss_dict
