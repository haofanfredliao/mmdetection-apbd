# Copyright (c) OpenMMLab. All rights reserved.
# V2: adds boundary aux loss on full-resolution decoder outputs and
#     per-image quality loss weighting via img_meta['loss_weight'].
from typing import Dict, List, Optional, Tuple

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
    """Extends Mask2FormerHead with several improvements for agricultural
    field segmentation, each independently toggleable via its own loss cfg
    so any combination can be stacked in a single training run:

    1. **Boundary aux loss** (``loss_boundary``, Track 2 / E4): computed at
       (capped) decoder feature resolution on morphologically-extracted
       boundary bands (:class:`BoundaryDiceLoss`).

    2. **Non-overlap loss** (``loss_nonoverlap``, Track 1 / E2): penalizes
       overlap among matched positive masks of the same image
       (:class:`NonOverlapLoss`).

    3. **Surface / Kervadec boundary loss** (``loss_surface``, Track 2 /
       E4b): GT signed-distance-map loss for direct boundary localization
       (:class:`KervadecBoundaryLoss`).

    4. **Curvature regularization** (``loss_curvature``, Track 3 / E5):
       penalizes jagged boundaries while protecting sharp GT corners
       (:class:`CurvatureLoss`).

    5. **Per-image quality weighting**: if ``img_meta`` contains a
       ``loss_weight`` key (float), each GT instance from that image is
       scaled by that factor when computing ``loss_dice`` and all of the
       auxiliary losses above. ``3_extreme`` images are excluded at the
       dataset level; ``2_lazy`` images typically carry a reduced weight.

    Only ``loss_dice`` and the auxiliary losses are quality-weighted; the
    cross-entropy ``loss_mask`` is left unscaled for training stability.

    All auxiliary losses are computed once, on the last decoder layer only,
    each at its own capped resolution (``*_max_res``) to bound memory/CPU
    cost — see the per-loss docstrings in ``mmdet/models/losses/
    boundary_loss.py`` for why (e.g. the surface loss's GT distance
    transform runs on CPU).

    New config args (beyond Mask2FormerHead):
        loss_boundary (dict, optional): Config for the boundary aux loss.
            Recommended: ``dict(type='BoundaryDiceLoss', loss_weight=2.0,
            kernel_size=3, eps=1e-5)``.
        loss_nonoverlap (dict, optional): Config for the non-overlap loss.
        loss_surface (dict, optional): Config for the Kervadec surface
            loss. Recommended starting weight is small (e.g. 0.01-0.1);
            see :class:`~mmdet.models.losses.KervadecBoundaryLoss`.
        surface_mode (str): Where the surface loss is evaluated.
            ``'dense'`` averages it over the whole downsampled frame at the
            last decoder layer only. ``'point'`` instead reads the distance
            map at the coordinates Mask2Former already importance-samples
            for dice/CE, and runs on every decoder layer. Prefer
            ``'point'``: under ``'dense'`` most of the frame sits deep
            inside or outside the mask where the sigmoid is saturated, so
            those pixels dominate the reported value while contributing
            almost no gradient.
        loss_curvature (dict, optional): Config for the curvature
            regularization loss.
        Any of the above left as ``None`` (the default) omits that loss
        term entirely — no combinatorics to worry about when enabling a
        subset.
    """

    # name -> (init kwarg for the loss cfg, init kwarg for its max_res,
    #          default max_res, loss_dict key)
    _AUX_LOSS_NAMES = ('boundary', 'nonoverlap', 'surface', 'curvature')

    def __init__(self,
                 loss_boundary=None,
                 boundary_max_res: int = 256,
                 loss_nonoverlap=None,
                 nonoverlap_max_res: int = 128,
                 loss_surface=None,
                 surface_max_res: int = 128,
                 surface_mode: str = 'dense',
                 loss_curvature=None,
                 curvature_max_res: int = 128,
                 **kwargs):
        super().__init__(**kwargs)
        assert surface_mode in ('dense', 'point'), \
            f"surface_mode must be 'dense' or 'point', got {surface_mode}"
        self.surface_mode = surface_mode
        self._gt_dist_cache = None
        self.loss_boundary = (
            MODELS.build(loss_boundary) if loss_boundary is not None else None)
        self.boundary_max_res = boundary_max_res
        self.loss_nonoverlap = (
            MODELS.build(loss_nonoverlap)
            if loss_nonoverlap is not None else None)
        self.nonoverlap_max_res = nonoverlap_max_res
        self.loss_surface = (
            MODELS.build(loss_surface) if loss_surface is not None else None)
        self.surface_max_res = surface_max_res
        self.loss_curvature = (
            MODELS.build(loss_curvature)
            if loss_curvature is not None else None)
        self.curvature_max_res = curvature_max_res

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def active_aux_loss_names(self) -> List[str]:
        """Names (subset of ``_AUX_LOSS_NAMES``) of the auxiliary losses
        that were configured (i.e. not None), in a fixed, deterministic
        order."""
        return [
            name for name in self._AUX_LOSS_NAMES
            if getattr(self, f'loss_{name}') is not None
        ]

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

    def _build_nonoverlap_group_weights(self, mask_targets_list,
                                        batch_img_metas, reference_tensor):
        """Build per-image weights for groups with at least two positives."""
        chunks = []
        for i, t in enumerate(mask_targets_list):
            if t.shape[0] <= 1:
                continue
            w = float(batch_img_metas[i].get('loss_weight', 1.0))
            chunks.append(reference_tensor.new_tensor(w))
        if not chunks:
            return None
        return torch.stack(chunks, dim=0)

    @staticmethod
    def _resize_pred(mask_preds_pos: Tensor, max_res: int) -> Tensor:
        """Downsample predicted mask logits to at most ``max_res`` per
        side (bilinear), leaving them untouched if already smaller."""
        h_feat, w_feat = mask_preds_pos.shape[-2:]
        out_h, out_w = min(h_feat, max_res), min(w_feat, max_res)
        if out_h < h_feat or out_w < w_feat:
            return F.interpolate(
                mask_preds_pos.unsqueeze(1), size=(out_h, out_w),
                mode='bilinear', align_corners=False).squeeze(1)
        return mask_preds_pos

    @staticmethod
    def _resize_target(mask_targets: Tensor, out_hw: Tuple[int, int]
                       ) -> Tensor:
        """Downsample a binary GT mask to ``out_hw`` (nearest, no grad)."""
        with torch.no_grad():
            return F.interpolate(
                mask_targets.unsqueeze(1).float(), size=out_hw,
                mode='nearest').squeeze(1)

    def _resize_pred_target(self, mask_preds_pos: Tensor,
                            mask_targets: Tensor, max_res: int
                            ) -> Tuple[Tensor, Tensor]:
        pred = self._resize_pred(mask_preds_pos, max_res)
        target = self._resize_target(mask_targets, pred.shape[-2:])
        return pred, target

    def _compute_aux_losses(self, aux_loss_names: List[str],
                            mask_preds_pos: Tensor, mask_targets: Tensor,
                            mask_targets_list, batch_img_metas,
                            per_gt_w, num_total_masks) -> Dict[str, Tensor]:
        """Compute every requested auxiliary loss and return a
        ``{name: loss_tensor}`` dict. Only called for the last decoder
        layer, and only for losses that were actually configured."""
        aux_losses: Dict[str, Tensor] = {}

        if 'boundary' in aux_loss_names:
            pred_bdr, target_bdr = self._resize_pred_target(
                mask_preds_pos, mask_targets, self.boundary_max_res)
            aux_losses['boundary'] = self.loss_boundary(
                pred_bdr, target_bdr, weight=per_gt_w,
                avg_factor=num_total_masks)

        if 'nonoverlap' in aux_loss_names:
            mask_preds_ov = self._resize_pred(
                mask_preds_pos, self.nonoverlap_max_res)
            group_sizes = [t.shape[0] for t in mask_targets_list]
            group_weights = self._build_nonoverlap_group_weights(
                mask_targets_list, batch_img_metas, mask_preds_pos)
            aux_losses['nonoverlap'] = self.loss_nonoverlap(
                mask_preds_ov, group_sizes, group_weights=group_weights,
                avg_factor=num_total_masks)

        # 'point' mode is handled in _loss_by_feat_single instead, since it
        # needs the sampled coordinates and runs on every decoder layer.
        if 'surface' in aux_loss_names and self.surface_mode == 'dense':
            pred_sfc, target_sfc = self._resize_pred_target(
                mask_preds_pos, mask_targets, self.surface_max_res)
            aux_losses['surface'] = self.loss_surface(
                pred_sfc, target_sfc, weight=per_gt_w,
                avg_factor=num_total_masks)

        if 'curvature' in aux_loss_names:
            pred_crv, target_crv = self._resize_pred_target(
                mask_preds_pos, mask_targets, self.curvature_max_res)
            aux_losses['curvature'] = self.loss_curvature(
                pred_crv, target_crv, weight=per_gt_w,
                avg_factor=num_total_masks)

        return aux_losses

    @property
    def _surface_on_points(self) -> bool:
        return self.loss_surface is not None and self.surface_mode == 'point'

    def _build_gt_distance_cache(self, batch_gt_instances) -> List[Tensor]:
        """Signed distance map of every GT mask in the batch, one entry per
        image, computed once per iteration.

        The distance transform runs on CPU via scipy and is by far the most
        expensive part of the surface loss; computing it inside
        ``_loss_by_feat_single`` would repeat it for all ten decoder layers
        and cost ~70% extra wall clock. The Hungarian assignment reorders the
        GT differently at each layer, but only ever *permutes* it — each GT is
        matched exactly once — so a layer's targets can be obtained by
        indexing this cache with that layer's ``pos_assigned_gt_inds``.
        Downsampling commutes with the permutation (it is per-mask), so the
        cache is built at the capped resolution directly.
        """
        cache = []
        for gt in batch_gt_instances:
            masks = gt.masks
            h, w = masks.shape[-2:]
            out_hw = (min(h, self.surface_max_res),
                      min(w, self.surface_max_res))
            if masks.shape[0] == 0:
                # Still needs the right spatial shape: images with no GT are
                # concatenated with the rest of the batch.
                cache.append(
                    masks.new_zeros((0, ) + out_hw, dtype=torch.float32))
                continue
            target_sfc = self._resize_target(masks, out_hw)
            cache.append(self.loss_surface.signed_distance_map(target_sfc))
        return cache

    def _surface_loss_on_points(self, sampling_results, points_coords: Tensor,
                                mask_point_preds: Tensor,
                                per_gt_w, num_total_masks) -> Tensor:
        """Kervadec surface loss read at Mask2Former's sampled coordinates."""
        from mmcv.ops import point_sample

        with torch.no_grad():
            dist_map = torch.cat([
                self._gt_dist_cache[i][res.pos_assigned_gt_inds]
                for i, res in enumerate(sampling_results)
            ], dim=0)
            phi_points = point_sample(
                dist_map.unsqueeze(1), points_coords).squeeze(1)

        return self.loss_surface.loss_from_signed_distance(
            mask_point_preds, phi_points, weight=per_gt_w,
            avg_factor=num_total_masks)

    # ------------------------------------------------------------------
    # Override _loss_by_feat_single
    # ------------------------------------------------------------------

    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor,
                             batch_gt_instances, batch_img_metas,
                             aux_loss_names: Optional[List[str]] = None
                             ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """Loss for a single decoder layer.

        Adds quality weighting via img_meta['loss_weight'].
        ``aux_loss_names`` lists which auxiliary losses to compute (only
        passed non-empty for the last decoder layer); an empty dict is
        returned as the 4th element otherwise.

        Returns:
            tuple: ``(loss_cls, loss_mask, loss_dice, aux_losses)`` where
            ``aux_losses`` is a ``{name: Tensor}`` dict (possibly empty).
        """
        from mmcv.ops import point_sample
        from ..utils import get_uncertain_point_coords_with_randomness

        aux_loss_names = aux_loss_names or []

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        # The point-sampled surface loss needs the Hungarian assignment to
        # index the per-iteration distance-map cache.
        if self._surface_on_points:
            (labels_list, label_weights_list, mask_targets_list,
             mask_weights_list, avg_factor,
             sampling_results) = self.get_targets(
                 cls_scores_list, mask_preds_list,
                 batch_gt_instances, batch_img_metas,
                 return_sampling_results=True)
        else:
            sampling_results = None
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
            aux_losses = {name: zero for name in aux_loss_names}
            if self._surface_on_points:
                aux_losses['surface'] = zero
            return loss_cls, zero, zero, aux_losses

        # --- per-gt quality weights ---
        per_gt_w = self._build_per_gt_quality_weights(
            mask_targets_list, batch_img_metas, mask_preds_pos)

        # ------------------------------------------------------------------
        # Auxiliary losses – each computed at its own capped resolution.
        # Only runs for the last decoder layer (aux_loss_names non-empty).
        # ------------------------------------------------------------------
        aux_losses = self._compute_aux_losses(
            aux_loss_names, mask_preds_pos, mask_targets, mask_targets_list,
            batch_img_metas, per_gt_w, num_total_masks)

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

        # Point-sampled surface loss rides on the coordinates drawn above and
        # is therefore computed for every decoder layer, not just the last.
        if self._surface_on_points:
            aux_losses['surface'] = self._surface_loss_on_points(
                sampling_results, points_coords, mask_point_preds,
                per_gt_w, num_total_masks)

        return loss_cls, loss_mask, loss_dice, aux_losses

    # ------------------------------------------------------------------
    # Override loss_by_feat
    # ------------------------------------------------------------------

    def loss_by_feat(self, all_cls_scores, all_mask_preds,
                     batch_gt_instances, batch_img_metas):
        """Compute losses across decoder layers.

        For all-but-last layers: standard cls/mask/dice losses only, plus
        the surface loss when it runs in ``'point'`` mode.
        For the last layer: additionally compute every configured
        auxiliary loss (once, each at its own capped resolution).
        """
        num_dec_layers = len(all_cls_scores)

        # Built once here rather than per layer; see _build_gt_distance_cache.
        if self._surface_on_points:
            self._gt_dist_cache = self._build_gt_distance_cache(
                batch_gt_instances)

        # Aux decoder layers (all but last) — no dense auxiliary losses.
        if num_dec_layers > 1:
            aux_cls, aux_mask, aux_dice, aux_extra = multi_apply(
                self._loss_by_feat_single,
                list(all_cls_scores[:-1]),
                list(all_mask_preds[:-1]),
                [batch_gt_instances] * (num_dec_layers - 1),
                [batch_img_metas] * (num_dec_layers - 1))
        else:
            aux_cls, aux_mask, aux_dice, aux_extra = [], [], [], []

        # Last decoder layer — with every configured auxiliary loss.
        active_names = self.active_aux_loss_names
        last_cls, last_mask, last_dice, last_aux = self._loss_by_feat_single(
            all_cls_scores[-1], all_mask_preds[-1],
            batch_gt_instances, batch_img_metas,
            aux_loss_names=active_names)

        loss_dict = dict()
        loss_dict['loss_cls'] = last_cls
        loss_dict['loss_mask'] = last_mask
        loss_dict['loss_dice'] = last_dice
        for name in active_names:
            loss_dict[f'loss_{name}'] = last_aux[name]

        for dec_i, (lc, lm, ld, extra) in enumerate(
                zip(aux_cls, aux_mask, aux_dice, aux_extra)):
            loss_dict[f'd{dec_i}.loss_cls'] = lc
            loss_dict[f'd{dec_i}.loss_mask'] = lm
            loss_dict[f'd{dec_i}.loss_dice'] = ld
            if 'surface' in extra:
                loss_dict[f'd{dec_i}.loss_surface'] = extra['surface']

        self._gt_dist_cache = None
        return loss_dict
