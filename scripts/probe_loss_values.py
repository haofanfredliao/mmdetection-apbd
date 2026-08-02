"""Compare two checkpoints on the quantities the auxiliary losses target.

A loss decreasing over training does not show that it did anything: dice falls
too, and sharper masks overlap less regardless of whether anything penalised
overlap. The test that separates the two is to evaluate the *same* objective
on a model that was never trained with it. This script runs both checkpoints
over identical batches and reports each auxiliary loss at unit weight, so the
numbers are directly comparable.

Run in eval mode so dropout does not add noise. Note the surface loss samples
each model's own uncertainty band, which is deliberate: that is the quantity
the loss actually optimises for that model.

Usage:
    python scripts/probe_loss_values.py CONFIG --checkpoints A.pth B.pth \
        [--labels v3 v5] [--num-batches 8] [--batch-size 2]
"""
import argparse
from collections import defaultdict

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from torch.utils.data import DataLoader

from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules

# Aux losses to report, and the head attribute owning each weight.
AUX = {'loss_nonoverlap': 'loss_nonoverlap', 'loss_surface': 'loss_surface'}
CONTEXT = ('loss_dice', 'loss_mask', 'loss_boundary')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--checkpoints', nargs=2, required=True)
    parser.add_argument('--labels', nargs=2, default=['A', 'B'])
    parser.add_argument('--num-batches', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument(
        '--surface-mode', choices=['point', 'dense', 'config'],
        default='config',
        help="How to evaluate the surface term. 'point' samples each model's "
             "own uncertainty band, which is NOT comparable across models: a "
             "model with a tighter band samples where |phi| is small and "
             "scores less negative for that reason alone. Use 'dense' to "
             "compare two checkpoints on the same fixed grid.")
    args = parser.parse_args()

    register_all_modules()
    init_default_scope('mmdet')
    cfg = Config.fromfile(args.config)

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    # shuffle=False so both checkpoints see byte-identical batches.
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
        collate_fn=pseudo_collate)
    batches = []
    it = iter(loader)
    for _ in range(args.num_batches):
        batches.append(next(it))

    results = {}
    for label, ckpt in zip(args.labels, args.checkpoints):
        model = MODELS.build(cfg.model)
        load_checkpoint(model, ckpt, map_location='cpu')
        model = model.cuda().eval()
        head = model.panoptic_head
        # Report at unit weight; the config values are only ramp start points.
        for attr in AUX.values():
            getattr(head, attr).loss_weight = 1.0
        if args.surface_mode != 'config':
            head.surface_mode = args.surface_mode

        acc = defaultdict(list)
        with torch.no_grad():
            for raw in batches:
                data = model.data_preprocessor(raw, True)
                losses = model.loss(data['inputs'], data['data_samples'])
                merged = defaultdict(float)
                for k, v in losses.items():
                    base = k.split('.')[-1]
                    merged[base] += float(v)
                for k in list(AUX) + list(CONTEXT):
                    if k in merged:
                        acc[k].append(merged[k])
        results[label] = {k: sum(v) / len(v) for k, v in acc.items()}
        del model
        torch.cuda.empty_cache()
        print(f'  {label} done', flush=True)

    a, b = args.labels
    print('\n' + '=' * 66)
    print(f'{"quantity":<20}{a:>13}{b:>13}{"change":>20}')
    print('-' * 66)
    for k in list(AUX) + list(CONTEXT):
        if k not in results[a] or k not in results[b]:
            continue
        va, vb = results[a][k], results[b][k]
        if abs(va) > 1e-12:
            pct = f'{(vb - va) / abs(va) * 100:+.1f}%'
        else:
            pct = 'n/a'
        print(f'{k:<20}{va:>13.4f}{vb:>13.4f}{pct:>20}')
    print('-' * 66)
    print(f'aux losses reported at unit weight; summed over decoder layers; '
          f'{args.num_batches} identical batches of {args.batch_size}')
    print('=' * 66)


if __name__ == '__main__':
    main()
