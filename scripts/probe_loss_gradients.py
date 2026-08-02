"""Measure how much gradient each loss term actually delivers.

Motivation: v4's surface loss was tuned by looking at its printed value and
turned out to be inert — the dense average was dominated by saturated pixels
that carry value but no gradient. Loss magnitude is therefore the wrong
quantity to size a weight by. This script backprops each term separately from
the same forward pass and reports the resulting gradient norm over the
trainable parameters, plus the norm per unit of loss_weight so a target
weight can be solved for directly.

Run against a trained checkpoint, not a random init: these auxiliary terms are
meant to act on an already roughly-correct mask, and their relative gradient
scale is completely different before that.

Usage:
    python scripts/probe_loss_gradients.py CONFIG --checkpoint CKPT \
        [--num-batches 3] [--target-ratio 0.1]
"""
import argparse
from collections import defaultdict

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules

# Terms whose weight we are trying to size, mapped to the loss module that
# owns the weight (so the probe can normalize by it).
PROBE_TERMS = {
    'loss_dice': None,           # reference term, weight left as configured
    'loss_mask': None,
    'loss_boundary': 'loss_boundary',
    'loss_nonoverlap': 'loss_nonoverlap',
    'loss_surface': 'loss_surface',
}


def grad_norm_of(loss, params):
    grads = torch.autograd.grad(
        loss, params, retain_graph=True, allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += g.detach().float().pow(2).sum().item()
    return total ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--num-batches', type=int, default=3)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Smaller than training by default: holding the graph for five '
             'separate backward passes does not fit at the training batch '
             'size, and the gradient ratios this probe reports are scale '
             'invariant.')
    parser.add_argument(
        '--target-ratio',
        type=float,
        default=0.1,
        help='Desired gradient norm of each aux term relative to loss_dice; '
             'the suggested weight is solved from this.')
    args = parser.parse_args()

    register_all_modules()
    init_default_scope('mmdet')
    cfg = Config.fromfile(args.config)

    model = MODELS.build(cfg.model)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().train()

    # Build the train loader directly so the batch matches training exactly
    # (same pipeline, same quality filtering, same batch size).
    loader_cfg = cfg.train_dataloader
    dataset = DATASETS.build(loader_cfg.dataset)
    from mmengine.dataset import pseudo_collate
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=pseudo_collate)

    params = [p for p in model.parameters() if p.requires_grad]

    values = defaultdict(list)
    norms = defaultdict(list)
    totals = []

    it = iter(loader)
    for b in range(args.num_batches):
        data = model.data_preprocessor(next(it), True)
        losses = model.loss(data['inputs'], data['data_samples'])

        # Sum the per-decoder-layer copies of each term so the probe reports
        # what the optimizer actually sees, not just the last layer's share.
        merged = defaultdict(lambda: 0.0)
        for k, v in losses.items():
            base = k.split('.')[-1]
            if base in PROBE_TERMS:
                merged[base] = merged[base] + v

        total = sum(
            v for k, v in losses.items() if isinstance(v, torch.Tensor))
        totals.append(float(total))

        for name, loss in merged.items():
            values[name].append(float(loss))
            norms[name].append(grad_norm_of(loss, params))
        print(f'  batch {b + 1}/{args.num_batches} done', flush=True)
        model.zero_grad(set_to_none=True)

    def mean(xs):
        return sum(xs) / len(xs)

    head = model.panoptic_head
    ref = mean(norms['loss_dice'])

    print('\n' + '=' * 78)
    print(f'{"term":<18}{"value":>11}{"grad_norm":>13}'
          f'{"vs dice":>10}{"weight":>9}{"suggest":>10}')
    print('-' * 78)
    for name in PROBE_TERMS:
        if name not in norms:
            continue
        gn = mean(norms[name])
        attr = PROBE_TERMS[name]
        w = getattr(getattr(head, attr), 'loss_weight', None) if attr else None
        if w:
            # grad norm is linear in loss_weight, so the weight hitting
            # target_ratio follows directly.
            suggest = f'{args.target_ratio * ref / (gn / w):.4g}'
            wtxt = f'{w:g}'
        else:
            suggest, wtxt = '-', '-'
        print(f'{name:<18}{mean(values[name]):>11.4f}{gn:>13.4f}'
              f'{gn / ref:>10.3f}{wtxt:>9}{suggest:>10}')
    print('-' * 78)
    print(f'total loss (all terms): {mean(totals):.4f}')
    print(f'suggested weights target grad_norm = {args.target_ratio:g} '
          f'x loss_dice grad_norm')
    print('=' * 78)


if __name__ == '__main__':
    main()
