# Copyright (c) OpenMMLab. All rights reserved.
# PolyFormer detector: instance segmentation via polygon vertex prediction.
# Visual-only adaptation of PolyFormer (CVPR 2023) for mmdetection.
from typing import Dict, List, Tuple, Union

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector


@MODELS.register_module()
class PolyFormer(SingleStageDetector):
    """PolyFormer instance segmentation detector.

    A visual-only adaptation of:
        "PolyFormer: Referring Image Segmentation as Sequential Polygon
        Generation" (Liu et al., CVPR 2023)

    The model predicts instance polygons (sequences of vertices) instead of
    pixel-level masks, making it especially well-suited for agricultural field
    boundary detection where parcel boundaries are inherently polygonal.

    Architecture::

        Image → Backbone (Swin-B/L) → FPN Neck
              → PolyFormerInsHead
                  ├─ Visual encoder  (PolyFormer-style rel-pos biases)
                  ├─ DETR decoder    (parallel object queries)
                  ├─ Classification head   (query → class label)
                  └─ Polygon head    (query → V×2 vertex coords)

    The key innovation inherited from PolyFormer is the **bilinear coordinate
    representation**: predicted vertex coordinates are refined by sampling
    visual features at sub-grid precision via bilinear interpolation, yielding
    sharper boundary polygon predictions.

    Args:
        backbone (ConfigType): Backbone config (e.g. SwinTransformer).
        neck (ConfigType, optional): Neck config (e.g. FPN).
        bbox_head (ConfigType): Head config (PolyFormerInsHead).
        train_cfg (ConfigType, optional): Training config.
        test_cfg (ConfigType, optional): Testing config.
        data_preprocessor (ConfigType, optional): Data preprocessor config.
        init_cfg (ConfigType, optional): Init config.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Run inference and add predictions to data samples.

        Args:
            batch_inputs: (B, C, H, W) preprocessed images.
            batch_data_samples: Data samples with metainfo.
            rescale: Rescale predictions to original image resolution.

        Returns:
            Updated SampleList with ``pred_instances`` containing:
                - scores (Tensor)
                - labels (Tensor)
                - bboxes (Tensor): xyxy in pixels
                - masks  (Tensor): (N, H, W) bool
                - polygons (list[ndarray]): (V, 2) pixel-space vertices
        """
        feats = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            feats, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
