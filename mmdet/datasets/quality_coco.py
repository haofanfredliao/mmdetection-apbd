"""Dataset wrapper that applies quality-based filtering and loss weighting.

Reads quality labels from a CSV file alongside a COCO annotation JSON and:
  - **Excludes** images labelled ``3_extreme`` from training/validation.
  - Sets ``loss_weight = 0.2`` for ``2_lazy`` images (passed through
    ``img_meta`` to the model head for per-gt sample weighting).
  - Sets ``loss_weight = 1.0`` for all other images.

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

from mmdet.registry import DATASETS
from .coco import CocoDataset

_QUALITY_EXCLUDE = '3_extreme'
_QUALITY_LAZY = '2_lazy'


@DATASETS.register_module()
class QualityAwareCocoDataset(CocoDataset):
    """CocoDataset that filters and weights samples by quality label.

    Args:
        quality_csv (str): Path to the quality report CSV file.
        lazy_loss_weight (float): Loss weight applied to ``2_lazy`` samples.
            Defaults to 0.2.
        **kwargs: Forwarded to :class:`CocoDataset`.
    """

    def __init__(self, quality_csv: str, lazy_loss_weight: float = 0.2,
                 **kwargs):
        self._quality_csv_path = quality_csv
        self._lazy_loss_weight = lazy_loss_weight
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

    def load_data_list(self):
        """Load and filter the COCO data list, attaching loss weights."""
        data_list = super().load_data_list()

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
