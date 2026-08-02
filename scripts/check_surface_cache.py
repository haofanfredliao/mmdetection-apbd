"""Verify the per-iteration distance-map cache is indexed correctly.

The point-sampled surface loss computes each GT's signed distance map once per
iteration and then reorders it per decoder layer with that layer's
``pos_assigned_gt_inds``. If that indexing were off, training would silently
optimise each mask against another parcel's distance map. This checks, for
every decoder layer of a real batch, that the reordered cache is bit-identical
to recomputing the transform from the layer's own matched ``mask_targets``.

Usage:
    python scripts/check_surface_cache.py CONFIG [--checkpoint CKPT]
"""
import argparse

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from torch.utils.data import DataLoader

from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()

    register_all_modules()
    init_default_scope('mmdet')
    cfg = Config.fromfile(args.config)

    model = MODELS.build(cfg.model)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().train()
    head = model.panoptic_head
    assert head.surface_mode == 'point', 'config is not in point mode'

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
        collate_fn=pseudo_collate)

    # Capture the matched targets of each layer, and the distance map the
    # cache produced for that same layer.
    seen_targets = []
    seen_dist = []

    orig_targets = head.get_targets
    orig_loss = head.loss_surface.loss_from_signed_distance

    def spy_targets(*a, **kw):
        out = orig_targets(*a, **kw)
        seen_targets.append(torch.cat(out[2], dim=0))
        return out

    # The loss only ever sees phi already sampled at the query points, so the
    # dense reordered map has to be caught on its way into point_sample. It is
    # the last thing sampled before the loss is called.
    import mmcv.ops
    orig_point_sample = mmcv.ops.point_sample
    last_sampled = []

    def spy_point_sample(inp, *a, **kw):
        last_sampled.append(inp)
        return orig_point_sample(inp, *a, **kw)

    def spy_loss(pred, dist_map, *a, **kw):
        seen_dist.append(last_sampled[-1].squeeze(1))
        return orig_loss(pred, dist_map, *a, **kw)

    head.get_targets = spy_targets
    head.loss_surface.loss_from_signed_distance = spy_loss
    mmcv.ops.point_sample = spy_point_sample

    data = model.data_preprocessor(next(iter(loader)), True)
    model.loss(data['inputs'], data['data_samples'])

    assert len(seen_dist) == len(seen_targets), (
        f'{len(seen_dist)} surface calls vs {len(seen_targets)} layers')

    print(f'checking {len(seen_dist)} decoder layers')
    ok = True
    for i, (targets, cached) in enumerate(zip(seen_targets, seen_dist)):
        h, w = targets.shape[-2:]
        out_hw = (min(h, head.surface_max_res), min(w, head.surface_max_res))
        expected = head.loss_surface.signed_distance_map(
            head._resize_target(targets, out_hw))
        if expected.shape != cached.shape:
            print(f'  layer {i}: shape expected {tuple(expected.shape)} '
                  f'vs cached {tuple(cached.shape)}; '
                  f'raw targets {tuple(targets.shape)}')
        same = expected.shape == cached.shape and torch.equal(expected, cached)
        ok &= same
        # Also report how many rows would have been wrong, to distinguish a
        # genuine permutation bug from an off-by-one in resizing.
        if not same and expected.shape == cached.shape:
            bad = (expected != cached).flatten(1).any(1).sum().item()
            extra = f'  ({bad}/{expected.shape[0]} masks differ)'
        else:
            extra = ''
        print(f'  layer {i}: n={targets.shape[0]:3d}  '
              f'{"OK" if same else "MISMATCH"}{extra}')

    print('\nPASS' if ok else '\nFAIL')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
