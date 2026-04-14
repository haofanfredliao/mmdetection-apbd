# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from PolyFormer (https://github.com/amazon-science/polygon-transformer)
# "PolyFormer: Referring Image Segmentation as Sequential Polygon Generation"
# Key adaption: language branch removed; transformed to DETR-style multi-instance
# segmentation with PolyFormer's bilinear vertex coordinate representation.
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import BaseModule, ModuleList, constant_init, xavier_init
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
import numpy as np

from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptMultiConfig, reduce_mean)

try:
    from skimage.draw import polygon as skimage_polygon
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


class MLP(BaseModule):
    """Multi-Layer Perceptron used for polygon coordinate regression.

    Mirrors the MLP in PolyFormer's decoder output head.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PolyFormerTransformerEncoder(BaseModule):
    """Transformer encoder for image patches.

    Takes projected FPN feature tokens and encodes them with
    relative positional biases following PolyFormer's design.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 ffn_ratio: float = 4.0,
                 dropout: float = 0.0,
                 image_bucket_size: int = 42,
                 attn_scale_factor: float = 2.0):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.image_bucket_size = image_bucket_size
        self.pos_scaling = (
            embed_dims / num_heads * attn_scale_factor) ** -0.5

        # Relative position bias tables for image patches
        image_num_rel_dis = (
            (2 * image_bucket_size - 1) * (2 * image_bucket_size - 1) + 3)
        self.image_rel_pos_tables = nn.ModuleList([
            nn.Embedding(image_num_rel_dis, num_heads)
            for _ in range(num_layers)
        ])
        for table in self.image_rel_pos_tables:
            nn.init.constant_(table.weight, 0)

        # Learnable absolute position projections
        self.image_pos_embed = nn.Embedding(image_bucket_size ** 2 + 1,
                                            embed_dims)
        self.image_pos_ln = nn.LayerNorm(embed_dims)
        self.pos_q_linear = nn.Linear(embed_dims, embed_dims)
        self.pos_k_linear = nn.Linear(embed_dims, embed_dims)

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dims,
                nhead=num_heads,
                dim_feedforward=int(embed_dims * ffn_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dims)

        # Pre-compute relative position bucket for image patches
        self._build_image_rp_bucket(image_bucket_size, image_num_rel_dis)

    def _build_image_rp_bucket(self, bucket_size: int,
                                num_rel_dis: int) -> None:
        coords_h = torch.arange(bucket_size)
        coords_w = torch.arange(bucket_size)
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, H, W
        coords_flatten = torch.flatten(coords, 1)  # 2, H*W
        relative_coords = (coords_flatten[:, :, None] -
                           coords_flatten[:, None, :])  # 2, H*W, H*W
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += bucket_size - 1
        relative_coords[:, :, 1] += bucket_size - 1
        relative_coords[:, :, 0] *= 2 * bucket_size - 1
        rp_bucket = torch.zeros(
            size=(bucket_size * bucket_size + 1,) * 2,
            dtype=relative_coords.dtype)
        rp_bucket[1:, 1:] = relative_coords.sum(-1)
        rp_bucket[0, 0:] = num_rel_dis - 3
        rp_bucket[0:, 0] = num_rel_dis - 2
        rp_bucket[0, 0] = num_rel_dis - 1
        self.register_buffer('image_rp_bucket', rp_bucket)

    def get_image_rel_pos_bias(self, image_pos_ids: Tensor,
                               layer_idx: int) -> Tensor:
        """Compute relative position bias matrix for image tokens."""
        bsz, seq_len = image_pos_ids.shape
        rp_size = self.image_rp_bucket.size(1)
        rp_bucket = (self.image_rp_bucket.unsqueeze(0).expand(
            bsz, rp_size, rp_size).gather(
                1,
                image_pos_ids[:, :, None].expand(bsz, seq_len,
                                                  rp_size)).gather(
                    2,
                    image_pos_ids[:, None, :].expand(bsz, seq_len, seq_len)))
        values = F.embedding(rp_bucket,
                             self.image_rel_pos_tables[layer_idx].weight)
        # (B, seq_len, seq_len, num_heads) → (B*num_heads, seq_len, seq_len)
        values = values.permute(0, 3, 1, 2).reshape(-1, seq_len, seq_len)
        return values

    def forward(self,
                x: Tensor,
                image_pos_ids: Tensor,
                key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Args:
            x: (B, N, C) image patch tokens
            image_pos_ids: (B, N) position indices for each patch
            key_padding_mask: (B, N) boolean mask (True = ignore)

        Returns:
            (B, N, C) encoded tokens
        """
        bsz, seq_len, _ = x.shape

        # Compute absolute position bias for attention
        pos_embed = self.image_pos_ln(self.image_pos_embed(image_pos_ids))
        pos_q = (self.pos_q_linear(pos_embed).view(
            bsz, seq_len, self.num_heads, -1).transpose(1, 2) *
                 self.pos_scaling)  # (B, H, N, d_head)
        pos_k = self.pos_k_linear(pos_embed).view(
            bsz, seq_len, self.num_heads, -1).transpose(1, 2)
        abs_pos_bias = torch.matmul(pos_q,
                                    pos_k.transpose(2, 3))  # (B, H, N, N)

        for idx, layer in enumerate(self.layers):
            # Combine absolute and relative position biases
            rel_bias = self.get_image_rel_pos_bias(
                image_pos_ids, idx)  # (B*H, N, N)
            attn_bias = abs_pos_bias.reshape(-1, seq_len, seq_len) + rel_bias

            # nn.TransformerEncoderLayer doesn't natively accept attn_bias
            # so we go through self-attention manually.
            x = self._layer_with_bias(layer, x, attn_bias, key_padding_mask)

        return self.norm(x)

    def _layer_with_bias(self, layer: nn.TransformerEncoderLayer, x: Tensor,
                         attn_bias: Tensor,
                         key_padding_mask: Optional[Tensor]) -> Tensor:
        """Run one TransformerEncoderLayer injecting additive attention bias."""
        # Pre-norm
        residual = x
        x_norm = layer.norm1(x)
        # Self-attention with bias
        attn_out, _ = layer.self_attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_bias,
            need_weights=False)
        x = residual + layer.dropout1(attn_out)
        # FFN
        residual = x
        x = residual + layer.dropout2(layer.linear2(
            layer.dropout(layer.activation(layer.linear1(layer.norm2(x))))))
        return x


class PolyFormerTransformerDecoder(BaseModule):
    """Transformer decoder that produces per-instance polygon queries.

    Follows the standard DETR decoder with cross-attention and
    PolyFormer's position-aware cross-attention biases.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 ffn_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dims,
                nhead=num_heads,
                dim_feedforward=int(embed_dims * ffn_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Args:
            tgt: (B, Q, C) object query tokens
            memory: (B, N, C) encoder output (image tokens)
            memory_key_padding_mask: (B, N) boolean mask

        Returns:
            (B, Q, C) decoded query tokens
        """
        x = tgt
        for layer in self.layers:
            x = layer(
                x,
                memory,
                memory_key_padding_mask=memory_key_padding_mask)
        return self.norm(x)


@MODELS.register_module()
class PolyFormerInsHead(BaseModule):
    """PolyFormer-based instance segmentation head.

    Adapted from:
        "PolyFormer: Referring Image Segmentation as Sequential Polygon
        Generation" (Liu et al., CVPR 2023)
        https://arxiv.org/abs/2302.07387

    Key differences from the original:
    - Language branch removed; pure vision-based instance segmentation.
    - Autoregressive decoding replaced with parallel DETR-style object queries.
    - PolyFormer's bilinear coordinate representation preserved: vertex features
      are sampled from image tokens via differentiable bilinear interpolation,
      enabling sub-grid coordinate precision.
    - Hungarian matching used for training (standard DETR practice).

    Architecture:
        1. Project multi-scale FPN features to ``embed_dims`` via 1×1 conv.
        2. Flatten spatial features → visual tokens for the encoder.
        3. PolyFormer encoder (with image relative positional biases).
        4. Transformer decoder with ``num_queries`` learnable object queries.
        5. Per-query prediction:
            - Class logits  (num_classes + 1 background)
            - Polygon vertices  (num_vertices × 2, normalized [0, 1])
        6. At inference: polygon rasterized to binary mask.

    Args:
        num_classes (int): Number of foreground classes (excl. background).
        in_channels (list[int]): Input feature-map channel counts from FPN.
        embed_dims (int): Transformer embedding dimension.
        num_queries (int): Number of instance object queries per image.
        num_vertices (int): Number of polygon vertices per instance.
        encoder_cfg (dict): Config for the visual encoder.
        decoder_cfg (dict): Config for the transformer decoder.
        loss_cls (dict): Config for classification loss.
        loss_poly (dict): Config for polygon L1 loss.
        train_cfg (dict, optional): Training config (ignored; kept for API).
        test_cfg (dict, optional): Testing config.  Keys:
            - ``score_thr`` (float): Score threshold for keeping instances.
        init_cfg (dict|list, optional): Weight init config.
    """

    def __init__(
            self,
            num_classes: int,
            in_channels: List[int],
            embed_dims: int = 256,
            num_queries: int = 100,
            num_vertices: int = 64,
            encoder_cfg: ConfigType = dict(
                num_layers=6,
                num_heads=8,
                ffn_ratio=4.0,
                dropout=0.1,
                image_bucket_size=42,
                attn_scale_factor=2.0),
            decoder_cfg: ConfigType = dict(num_layers=6,
                                           num_heads=8,
                                           ffn_ratio=4.0,
                                           dropout=0.1),
            loss_cls: ConfigType = dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=2.0,
                class_weight=None),
            loss_poly: ConfigType = dict(type='L1Loss', loss_weight=5.0),
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = dict(score_thr=0.5, max_per_img=100),
            init_cfg: OptMultiConfig = None,
            **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_queries = num_queries
        self.num_vertices = num_vertices
        self.test_cfg = test_cfg

        # Feature projection: one conv per FPN level, we use only the
        # finest level (index 0) for the main image tokens, coarser levels
        # are fused via simple add-after-project.
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                Conv2d(c, embed_dims, kernel_size=1),
                nn.GroupNorm(32, embed_dims),
            ) for c in in_channels
        ])

        # Visual encoder (PolyFormer-style with rel-pos biases)
        self.encoder = PolyFormerTransformerEncoder(
            embed_dims=embed_dims, **encoder_cfg)

        # DETR decoder
        self.decoder = PolyFormerTransformerDecoder(
            embed_dims=embed_dims, **decoder_cfg)

        # Learnable object queries
        self.query_feat = nn.Embedding(num_queries, embed_dims)
        self.query_pos = nn.Embedding(num_queries, embed_dims)

        # Prediction heads
        self.cls_head = Linear(embed_dims, num_classes + 1)
        self.poly_head = MLP(embed_dims, embed_dims, num_vertices * 2, 3)

        # Bilinear vertex feature sampling: project 2-channel (x,y) coords
        # to embed_dims for vertex refinement (PolyFormer's key contribution).
        self.vertex_proj = nn.Linear(embed_dims, embed_dims)

        # Losses
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_poly = MODELS.build(loss_poly)

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    def _prepare_visual_tokens(
            self,
            feats: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor, Tensor]:
        """Project FPN features to tokens for the encoder.

        For simplicity we project all FPN levels to ``embed_dims`` and
        use only the finest level (C2) as image tokens.  Coarser levels
        are spatially upsampled and added as additional context via the
        encoder's key/value.

        Returns:
            tokens: (B, H*W, C) flattened image tokens
            image_pos_ids: (B, H*W) integer position indices
            padding_mask: None (no padding required for dense features)
        """
        # Use only the finest FPN level (index 0, e.g. stride 4 or 8)
        feat = self.input_proj[0](feats[0])  # (B, C, H, W)

        # Fuse coarser levels
        for i in range(1, len(feats)):
            coarse = self.input_proj[i](feats[i])
            coarse_up = F.interpolate(
                coarse, size=feat.shape[-2:], mode='bilinear',
                align_corners=False)
            feat = feat + coarse_up

        B, C, H, W = feat.shape

        # Build 2-D position indices (mapped to 1-D index for embedding table)
        bucket = self.encoder.image_bucket_size
        h_idx = torch.arange(H, device=feat.device)
        w_idx = torch.arange(W, device=feat.device)
        pos_h = (h_idx.float() / H * (bucket - 1)).long()
        pos_w = (w_idx.float() / W * (bucket - 1)).long()
        pos_2d = pos_h.unsqueeze(1) * bucket + pos_w.unsqueeze(0) + 1
        image_pos_ids = pos_2d.view(1, -1).expand(B, -1)  # (B, H*W)

        # Flatten spatial dims
        tokens = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        return tokens, image_pos_ids, H, W

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
            self,
            feats: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            feats: Multi-scale features from FPN/backbone,
                   ordered from fine to coarse.

        Returns:
            cls_scores: (B, Q, num_classes+1)
            poly_preds: (B, Q, num_vertices*2) normalised coords in [0,1]
            vis_tokens: (B, N, C) visual tokens (for potential vertex refine)
        """
        tokens, image_pos_ids, H, W = self._prepare_visual_tokens(feats)
        B = tokens.shape[0]

        # Encode visual tokens
        memory = self.encoder(tokens, image_pos_ids)  # (B, N, C)

        # Decode: parallel object queries
        query_feat = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)
        tgt = query_feat + query_pos  # (B, Q, C)

        decode_out = self.decoder(tgt, memory)  # (B, Q, C)

        # Prediction heads
        cls_scores = self.cls_head(decode_out)  # (B, Q, num_classes+1)
        poly_preds = self.poly_head(decode_out).sigmoid()  # (B, Q, V*2)

        # --- PolyFormer bilinear vertex feature refinement ---
        # For each predicted vertex, sample the visual feature at that
        # location via bilinear interpolation, then use it to refine the
        # per-vertex representation.  This is the core geometric innovation
        # from PolyFormer (Sec. 3.2 bilinear token interpolation).
        poly_refined = self._bilinear_vertex_refine(
            poly_preds, memory, H, W)  # (B, Q, V*2)

        return cls_scores, poly_refined, memory

    def _bilinear_vertex_refine(self, poly_preds: Tensor, memory: Tensor,
                                H: int, W: int) -> Tensor:
        """Refine polygon vertex coordinates using bilinear image features.

        For each query's predicted vertices (x, y in [0,1]), we:
        1. Reshape memory back to (B, H, W, C) spatial feature map.
        2. Use F.grid_sample at the point locations to extract features.
        3. Project extracted features and add to the raw coordinate outputs.
           (residual refinement — the raw coords already carry the geometry,
            the refinement adds spatial detail from the image.)

        Args:
            poly_preds: (B, Q, V*2) initial coordinate predictions [0,1]
            memory: (B, H*W, C) encoded image tokens
            H, W: spatial dimensions of the feature map

        Returns:
            poly_refined: (B, Q, V*2) refined coordinates [0,1]
        """
        B, Q, V2 = poly_preds.shape
        V = V2 // 2

        # (B, C, H, W) feature map
        feat_map = memory.transpose(1, 2).view(B, -1, H, W)

        # Normalise to grid_sample's coordinate space [-1, 1]
        coords_xy = poly_preds.view(B, Q * V, 2)  # (B, Q*V, 2)
        grid = coords_xy * 2 - 1  # [0,1] → [-1,1]
        grid = grid.view(B, Q * V, 1, 2)  # (B, Q*V, 1, 2)

        # Sample features: output (B, C, Q*V, 1)
        sampled = F.grid_sample(
            feat_map, grid, mode='bilinear', align_corners=True,
            padding_mode='border')
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B, Q*V, C)
        sampled = sampled.view(B, Q, V, self.embed_dims)

        # Project to coordinate correction (B, Q, V, 2)
        delta = self.vertex_proj(sampled)  # (B, Q, V, C) — over-parameterised
        # Compress to 2D correction via a simple final layer
        delta = delta[..., :2]  # take first 2 dims as a shortcut
        delta = torch.tanh(delta) * 0.05  # small bounded refinement

        poly_refined = (poly_preds.view(B, Q, V, 2) + delta).clamp(0, 1)
        return poly_refined.view(B, Q, V2)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, feats: Tuple[Tensor, ...],
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Compute losses.

        Args:
            feats: FPN features.
            batch_data_samples: List of data samples with gt_instances.

        Returns:
            dict[str, Tensor]: loss components.
        """
        cls_scores, poly_preds, _ = self.forward(feats)
        return self.loss_by_feat(cls_scores, poly_preds, batch_data_samples)

    def loss_by_feat(self, cls_scores: Tensor, poly_preds: Tensor,
                     batch_data_samples: SampleList) -> Dict[str, Tensor]:
        batch_gt_instances = [s.gt_instances for s in batch_data_samples]
        batch_img_metas = [s.metainfo for s in batch_data_samples]

        # Hungarian matching
        indices = self._match(cls_scores.detach(), poly_preds.detach(),
                              batch_gt_instances, batch_img_metas)

        # --- Classification loss ---
        # Build target labels: background = num_classes
        cls_targets = cls_scores.new_full(
            (cls_scores.shape[0], cls_scores.shape[1]),
            self.num_classes,
            dtype=torch.long)
        for b, (pred_idx, gt_idx) in enumerate(indices):
            cls_targets[b, pred_idx] = batch_gt_instances[b].labels[gt_idx]

        loss_cls = self.loss_cls(
            cls_scores.view(-1, self.num_classes + 1),
            cls_targets.view(-1))

        # --- Polygon L1 loss (only on matched queries) ---
        num_matched = sum(len(pi) for pi, _ in indices)
        num_matched = max(reduce_mean(
            cls_scores.new_tensor([num_matched])), 1.0)

        loss_poly = cls_scores.new_zeros(1)
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            img_h, img_w = batch_img_metas[b]['img_shape'][:2]
            gt_polys = self._get_gt_polygons(
                batch_gt_instances[b], gt_idx,
                img_h, img_w)  # (K, V*2) normalised
            pred_polys = poly_preds[b][pred_idx]  # (K, V*2)
            loss_poly = loss_poly + self.loss_poly(
                pred_polys, gt_polys) * pred_polys.shape[0]

        loss_poly = loss_poly / num_matched

        return dict(loss_poly_cls=loss_cls, loss_poly_vertices=loss_poly)

    # ------------------------------------------------------------------
    # Hungarian matching helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _match(self, cls_scores: Tensor, poly_preds: Tensor,
               batch_gt_instances: InstanceList,
               batch_img_metas: List[dict]) -> List[Tuple]:
        """Per-image Hungarian matching between predictions and GT.

        Returns list of (pred_indices, gt_indices) tuples.
        """
        indices = []
        B = cls_scores.shape[0]
        for b in range(B):
            gt_inst = batch_gt_instances[b]
            num_gt = len(gt_inst)
            if num_gt == 0:
                indices.append(([], []))
                continue

            img_h, img_w = batch_img_metas[b]['img_shape'][:2]
            gt_polys = self._get_gt_polygons(
                gt_inst, list(range(num_gt)), img_h, img_w)  # (G, V*2)

            # Classification cost: negative probability of GT class
            cls_prob = cls_scores[b].softmax(-1)  # (Q, C+1)
            gt_labels = gt_inst.labels  # (G,)
            cost_cls = -cls_prob[:, gt_labels]  # (Q, G)

            # Polygon L1 cost
            pred = poly_preds[b]  # (Q, V*2)
            cost_poly = torch.cdist(
                pred.float(), gt_polys.float(), p=1)  # (Q, G)

            cost = cost_cls + cost_poly
            row_ind, col_ind = linear_sum_assignment(
                cost.cpu().numpy())
            indices.append((row_ind.tolist(), col_ind.tolist()))

        return indices

    def _get_gt_polygons(self, gt_inst: InstanceData, gt_idx: List[int],
                         img_h: int, img_w: int) -> Tensor:
        """Extract GT polygons and resample to ``num_vertices`` points.

        Handles two annotation formats:
        1. gt_inst.polygons: list[list[ndarray]] (COCO polygon format)
        2. gt_inst.masks: BitmapMasks or PolygonMasks

        Returns:
            (K, num_vertices*2) normalised coordinate tensor.
        """
        vertices_list = []
        for idx in gt_idx:
            verts = self._extract_polygon_from_gt(gt_inst, int(idx),
                                                   img_h, img_w)
            vertices_list.append(verts)

        if len(vertices_list) == 0:
            return gt_inst.labels.new_zeros(0, self.num_vertices * 2).float()

        return torch.stack(vertices_list, dim=0)  # (K, V*2)

    def _extract_polygon_from_gt(self, gt_inst: InstanceData, idx: int,
                                   img_h: int, img_w: int) -> Tensor:
        """Extract a single resampled + normalised polygon.

        Priority: polygons field → masks field (extract contour).
        Returns (num_vertices*2,) float tensor with values in [0,1].
        """
        device = self.query_feat.weight.device
        V = self.num_vertices

        # --- Try polygons annotation ---
        if hasattr(gt_inst, 'polygons') and gt_inst.polygons is not None:
            segs = gt_inst.polygons[idx]
            pts = self._largest_polygon(segs)  # (N, 2) in pixel coords
            if pts is not None:
                pts = self._resample_polygon(pts, V)  # (V, 2)
                pts[:, 0] = pts[:, 0] / img_w
                pts[:, 1] = pts[:, 1] / img_h
                return torch.tensor(
                    pts.reshape(-1), dtype=torch.float32, device=device)

        # --- Try masks annotation ---
        if hasattr(gt_inst, 'masks') and gt_inst.masks is not None:
            mask = gt_inst.masks[idx]
            if hasattr(mask, 'masks'):
                mask_np = mask.masks[0]  # BitmapMasks[idx]
            else:
                mask_np = mask  # numpy bool array
            pts = self._mask_to_polygon(mask_np)  # (N, 2)
            pts = self._resample_polygon(pts, V)  # (V, 2)
            pts[:, 0] = pts[:, 0] / img_w
            pts[:, 1] = pts[:, 1] / img_h
            return torch.tensor(
                pts.reshape(-1), dtype=torch.float32, device=device)

        # Fallback: zero polygon (should not happen in practice)
        return torch.zeros(V * 2, dtype=torch.float32, device=device)

    @staticmethod
    def _largest_polygon(segs) -> Optional['np.ndarray']:
        """Pick the largest polygon segment from a COCO annotation."""
        best = None
        best_len = 0
        for seg in segs:
            if isinstance(seg, (list, tuple)):
                arr = np.array(seg).reshape(-1, 2)
            else:
                arr = np.array(seg).reshape(-1, 2)
            if len(arr) > best_len:
                best_len = len(arr)
                best = arr
        return best

    @staticmethod
    def _mask_to_polygon(mask: 'np.ndarray') -> 'np.ndarray':
        """Convert binary mask to its contour polygon (largest cc)."""
        try:
            import cv2
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return np.zeros((4, 2), dtype=np.float32)
            cnt = max(contours, key=cv2.contourArea)
            return cnt.squeeze(1).astype(np.float32)
        except ImportError:
            # Fallback: bounding-box rectangle
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            return np.array([[cmin, rmin], [cmax, rmin],
                             [cmax, rmax], [cmin, rmax]], dtype=np.float32)

    @staticmethod
    def _resample_polygon(pts: 'np.ndarray', num: int) -> 'np.ndarray':
        """Uniformly resample polygon to exactly ``num`` vertices.

        Uses arc-length parameterisation + linear interpolation.
        """
        if len(pts) == 0:
            return np.zeros((num, 2), dtype=np.float32)
        if len(pts) == 1:
            return np.tile(pts, (num, 1)).astype(np.float32)

        # Close the polygon
        pts_closed = np.vstack([pts, pts[:1]])
        diffs = np.diff(pts_closed, axis=0)
        seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
        cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
        total_len = cum_len[-1]

        if total_len < 1e-6:
            return np.tile(pts[:1], (num, 1)).astype(np.float32)

        sample_dists = np.linspace(0, total_len, num, endpoint=False)
        new_pts = np.zeros((num, 2), dtype=np.float32)
        for i, d in enumerate(sample_dists):
            seg_idx = np.searchsorted(cum_len, d, side='right') - 1
            seg_idx = min(seg_idx, len(pts) - 1)
            t = (d - cum_len[seg_idx]) / max(seg_lens[seg_idx], 1e-8)
            p0 = pts[seg_idx % len(pts)]
            p1 = pts[(seg_idx + 1) % len(pts)]
            new_pts[i] = p0 + t * (p1 - p0)
        return new_pts

    # ------------------------------------------------------------------
    # Prediction / postprocessing
    # ------------------------------------------------------------------

    def predict(self, feats: Tuple[Tensor, ...],
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Run inference and return instance lists.

        Args:
            feats: FPN features.
            batch_data_samples: Data samples with metainfo.
            rescale: Whether to rescale polygons/masks to original image size.

        Returns:
            list[:obj:`InstanceData`]
        """
        cls_scores, poly_preds, _ = self.forward(feats)
        results_list = []
        for b in range(cls_scores.shape[0]):
            img_meta = batch_data_samples[b].metainfo
            result = self._predict_single(
                cls_scores[b], poly_preds[b], img_meta, rescale)
            results_list.append(result)
        return results_list

    def _predict_single(self, cls_score: Tensor, poly_pred: Tensor,
                        img_meta: dict, rescale: bool) -> InstanceData:
        """Post-process predictions for one image.

        Returns InstanceData with fields:
            - scores:  (N,) float confidence scores
            - labels:  (N,) int class indices
            - bboxes:  (N, 4) float xyxy bounding boxes (pixels)
            - masks:   (N, H, W) bool binary masks
            - polygons: list of (V, 2) numpy arrays (optional, pixel coords)
        """
        score_thr = self.test_cfg.get('score_thr', 0.5)
        max_per_img = self.test_cfg.get('max_per_img', 100)

        # Class probabilities (exclude background)
        probs = cls_score.softmax(-1)  # (Q, C+1)
        scores, labels = probs[:, :-1].max(-1)  # (Q,)

        # Filter by threshold
        keep = scores >= score_thr
        scores = scores[keep]
        labels = labels[keep]
        poly_pred = poly_pred[keep]

        # Top-k
        if scores.shape[0] > max_per_img:
            topk_idx = scores.topk(max_per_img)[1]
            scores = scores[topk_idx]
            labels = labels[topk_idx]
            poly_pred = poly_pred[topk_idx]

        result = InstanceData()
        result.scores = scores
        result.labels = labels

        if scores.shape[0] == 0:
            img_h, img_w = img_meta['img_shape'][:2]
            if rescale:
                img_h, img_w = img_meta.get('ori_shape', img_meta[
                    'img_shape'])[:2]
            result.bboxes = scores.new_zeros(0, 4)
            result.masks = scores.new_zeros(
                0, img_h, img_w).bool()
            return result

        # Determine output resolution
        img_h, img_w = img_meta['img_shape'][:2]
        if rescale:
            scale_factor = img_meta.get('scale_factor', (1.0, 1.0))
            ori_h, ori_w = img_meta.get('ori_shape', img_meta['img_shape'])[:2]
            out_h, out_w = ori_h, ori_w
        else:
            out_h, out_w = img_h, img_w

        N = poly_pred.shape[0]
        V = self.num_vertices
        polys_norm = poly_pred.view(N, V, 2).cpu().numpy()

        masks = np.zeros((N, out_h, out_w), dtype=bool)
        bboxes = np.zeros((N, 4), dtype=np.float32)
        poly_pixel_list = []

        for i, p in enumerate(polys_norm):
            px = (p[:, 0] * out_w).clip(0, out_w - 1)
            py = (p[:, 1] * out_h).clip(0, out_h - 1)
            poly_pixel_list.append(np.stack([px, py], axis=1))  # (V, 2)

            if _HAS_SKIMAGE:
                rr, cc = skimage_polygon(py, px, shape=(out_h, out_w))
                masks[i, rr, cc] = True
            else:
                # Fallback: bounding-box mask
                x1, y1 = int(px.min()), int(py.min())
                x2, y2 = int(px.max()), int(py.max())
                masks[i, y1:y2 + 1, x1:x2 + 1] = True

            m = masks[i]
            if m.any():
                rows = np.where(m.any(1))[0]
                cols = np.where(m.any(0))[0]
                bboxes[i] = [cols[0], rows[0], cols[-1], rows[-1]]
            else:
                bboxes[i] = [px.min(), py.min(), px.max(), py.max()]

        result.bboxes = scores.new_tensor(bboxes)
        result.masks = torch.from_numpy(masks).to(scores.device)
        result.polygons = poly_pixel_list
        return result
