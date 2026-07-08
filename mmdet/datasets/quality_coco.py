"""Dataset wrapper that applies quality-based filtering and loss weighting.

Reads quality labels from a CSV file alongside a COCO annotation JSON.

The default ``legacy`` mode preserves the original V2 behavior:
  - **Excludes** images labelled ``3_extreme`` from training/validation.
  - Sets ``loss_weight = 0.2`` for ``2_lazy`` images (passed through
    ``img_meta`` to the model head for per-gt sample weighting).
  - Sets ``loss_weight = 1.0`` for all other images.

For cleaned AI4Boundary experiments, ``good_with_background`` mode keeps all
``1_good`` images and samples true background negatives from ``2_lazy`` images
that have no COCO annotations. This excludes non-empty lazy labels.

CSV format (matching quality_report.csv)::

    split,sample_id,quality,...
    train,AT_10033,1_good,...
    train,AT_10034,2_lazy,...
    ...

The ``sample_id`` is the leading ``COUNTRY_NUMBER`` prefix of the image
filename, e.g. ``AT_10033`` maps to ``AT_10033_ortho_1m_512.png``.

Usage in config::

    train_dataloader = dict(
        dataset=dict(
            type='QualityAwareCocoDataset',
            quality_csv='data/data/ai4b_coco/quality_report.csv',
            lazy_loss_weight=0.2,
            ...
        )
    )

    # Also add 'loss_weight' to PackDetInputs meta_keys in the pipeline:
    train_pipeline = [
        ...,
        dict(type='PackDetInputs',
             meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                        'scale_factor', 'flip', 'flip_direction',
                        'loss_weight')),
    ]
"""
import csv
import os
import random

from mmdet.registry import DATASETS
from .coco import CocoDataset

_QUALITY_EXCLUDE = '3_extreme'
_QUALITY_LAZY = '2_lazy'


@DATASETS.register_module()
class QualityAwareCocoDataset(CocoDataset):
    """CocoDataset that filters and weights samples by quality label.

    Args:
        quality_csv (str): Path to the quality report CSV file.
        lazy_loss_weight (float): Loss weight applied to ``2_lazy`` samples in
            ``legacy`` mode. Defaults to 0.2.
        filter_mode (str): One of ``legacy``, ``good_only`` or
            ``good_with_background``. Defaults to ``legacy``.
        background_sample_ratio (float): In ``good_with_background`` mode,
            sample at most ``round(num_good * background_sample_ratio)``
            background-negative images. Defaults to 0.2.
        background_sample_seed (int): Deterministic seed for background
            sampling. Defaults to 42.
        **kwargs: Forwarded to :class:`CocoDataset`.
    """

    def __init__(self, quality_csv: str, lazy_loss_weight: float = 0.2,
                 filter_mode: str = 'legacy',
                 background_sample_ratio: float = 0.2,
                 background_sample_seed: int = 42,
                 **kwargs):
        self._quality_csv_path = quality_csv
        self._lazy_loss_weight = lazy_loss_weight
        self._filter_mode = filter_mode
        self._background_sample_ratio = background_sample_ratio
        self._background_sample_seed = background_sample_seed
        if filter_mode not in {
                'legacy', 'good_only', 'good_with_background'}:
            raise ValueError(
                'filter_mode must be one of "legacy", "good_only", or '
                f'"good_with_background", but got {filter_mode!r}.')
        # Build quality lookup BEFORE parent __init__ calls load_data_list
        self._quality_lookup = self._load_quality_csv(quality_csv)
        super().__init__(**kwargs)

    @staticmethod
    def _load_quality_csv(csv_path: str) -> dict:
        """Return a dict mapping sample_id -> quality string."""
        lookup = {}
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lookup[row['sample_id'].strip()] = row['quality'].strip()
        return lookup

    @staticmethod
    def _sample_id_from_filename(filename: str) -> str:
        """Extract leading ``COUNTRY_NUMBER`` prefix from filename.

        e.g. ``AT_10033_ortho_1m_512.png`` -> ``AT_10033``
        """
        basename = os.path.splitext(os.path.basename(filename))[0]
        # sample_id is everything up to (but not including) the third '_'
        # e.g.  AT_10033_ortho... -> split on _ => ['AT','10033','ortho',...]
        parts = basename.split('_')
        if len(parts) >= 2:
            return '_'.join(parts[:2])
        return basename

    def _get_loss_weight(self, filename: str) -> float:
        """Return the loss weight for a given image filename, or None if the
        image should be excluded entirely."""
        sample_id = self._sample_id_from_filename(filename)
        quality = self._quality_lookup.get(sample_id, '1_good')
        if quality == _QUALITY_EXCLUDE:
            return None  # signal to drop this image
        if quality == _QUALITY_LAZY:
            return self._lazy_loss_weight
        return 1.0

    def _quality_for_data_info(self, data_info: dict) -> str:
        filename = data_info.get('img_path', '')
        sample_id = self._sample_id_from_filename(filename)
        return self._quality_lookup.get(sample_id, '1_good')

    @staticmethod
    def _has_annotations(data_info: dict) -> bool:
        return len(data_info.get('instances', [])) > 0

    def _sample_background_negatives(self, background_infos: list,
                                     num_good: int) -> list:
        """Deterministically sample background negatives for clean training."""
        if self._background_sample_ratio is None:
            return background_infos
        max_background = int(round(num_good * self._background_sample_ratio))
        if max_background >= len(background_infos):
            return background_infos

        rng = random.Random(self._background_sample_seed)
        sampled = rng.sample(background_infos, max_background)
        sampled_ids = {id(info) for info in sampled}
        # Preserve original dataset order for reproducibility in logs/samplers.
        return [info for info in background_infos if id(info) in sampled_ids]

    def _load_clean_data_list(self, data_list: list) -> list:
        """Keep good positives plus sampled true background negatives."""
        good_infos = []
        background_infos = []
        dropped_lazy_positive = 0
        dropped_extreme = 0
        dropped_other = 0

        for data_info in data_list:
            quality = self._quality_for_data_info(data_info)
            has_ann = self._has_annotations(data_info)

            if quality == '1_good':
                data_info['loss_weight'] = 1.0
                good_infos.append(data_info)
            elif quality == _QUALITY_LAZY and not has_ann:
                data_info['loss_weight'] = 1.0
                background_infos.append(data_info)
            elif quality == _QUALITY_LAZY:
                dropped_lazy_positive += 1
            elif quality == _QUALITY_EXCLUDE:
                dropped_extreme += 1
            else:
                dropped_other += 1

        sampled_background = self._sample_background_negatives(
            background_infos, len(good_infos))
        good_ids = {id(info) for info in good_infos}
        sampled_ids = {id(info) for info in sampled_background}
        dropped_background = len(background_infos) - len(sampled_background)
        filtered = [
            info for info in data_list
            if id(info) in good_ids or id(info) in sampled_ids
        ]

        from mmengine.logging import MMLogger
        logger = MMLogger.get_current_instance()
        logger.info(
            f'QualityAwareCocoDataset[{self._filter_mode}]: loaded '
            f'{len(data_list)} images, kept {len(filtered)} '
            f'({len(good_infos)} good, {len(sampled_background)} background), '
            f'dropped {len(data_list) - len(filtered)} '
            f'(lazy_positive={dropped_lazy_positive}, '
            f'unsampled_background={dropped_background}, '
            f'extreme={dropped_extreme}, other={dropped_other}).')
        return filtered

    def _load_good_only_data_list(self, data_list: list) -> list:
        filtered = []
        for data_info in data_list:
            if self._quality_for_data_info(data_info) == '1_good':
                data_info['loss_weight'] = 1.0
                filtered.append(data_info)

        from mmengine.logging import MMLogger
        logger = MMLogger.get_current_instance()
        logger.info(
            f'QualityAwareCocoDataset[good_only]: loaded {len(data_list)} '
            f'images, kept {len(filtered)} good images, dropped '
            f'{len(data_list) - len(filtered)}.')
        return filtered

    def load_data_list(self):
        """Load and filter the COCO data list, attaching loss weights."""
        data_list = super().load_data_list()

        if self._filter_mode == 'good_with_background':
            return self._load_clean_data_list(data_list)
        if self._filter_mode == 'good_only':
            return self._load_good_only_data_list(data_list)

        filtered = []
        for data_info in data_list:
            filename = data_info.get('img_path', '')
            loss_weight = self._get_loss_weight(filename)
            if loss_weight is None:
                continue  # drop 3_extreme
            data_info['loss_weight'] = loss_weight
            filtered.append(data_info)

        total = len(data_list)
        kept = len(filtered)
        dropped = total - kept
        from mmengine.logging import MMLogger
        logger = MMLogger.get_current_instance()
        logger.info(
            f'QualityAwareCocoDataset: loaded {total} images, '
            f'dropped {dropped} (3_extreme), kept {kept} '
            f'({sum(1 for d in filtered if d["loss_weight"] < 1.0)} lazy, '
            f'{sum(1 for d in filtered if d["loss_weight"] == 1.0)} good).')
        return filtered
