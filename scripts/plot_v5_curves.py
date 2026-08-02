#!/usr/bin/env python3
"""Plot v5 training curves: loss groups + eval metric groups.

Reads logs/v5_train.log and writes figures to docs/figures/v5_curves/.
Similar-meaning quantities share a panel; layered terms are shown for a
representative subset of decoder layers (first / mid / late / final).
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG = Path('logs/v5_train.log')
OUT = Path('docs/figures/v5_curves')
MAX_ITERS = 16500
RAMP_END = int(MAX_ITERS * 0.1)  # LossWeightRampHook end for surface / nonoverlap

# Subset of decoder layers to keep the layered panels readable.
# Final-layer keys have no "dN." prefix in mmengine logs.
LAYER_KEYS = {
    'd0': 'd0',   # before first decoder layer
    'd4': 'd4',   # mid
    'd8': 'd8',   # late aux
    'final': '',  # last decoder layer (unprefixed)
}


def parse_kv(segment: str) -> dict:
    return {k: float(v) for k, v in re.findall(
        r'([\w./\-]+):\s+([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)', segment)}


def load_log(path: Path):
    train, val = [], []
    for line in open(path, errors='ignore'):
        m = re.search(r'Iter\(train\) \[ *(\d+)/', line)
        if m:
            d = parse_kv(line.split('Iter(train)', 1)[1])
            d['iter'] = int(m.group(1))
            train.append(d)
            continue
        if 'Iter(val) [187/187]' in line:
            d = parse_kv(line.split('Iter(val)', 1)[1])
            # val has no iter in the summary line; infer from count later
            val.append(d)
    # val every 660 iters
    for i, d in enumerate(val):
        d['iter'] = (i + 1) * 660
    return train, val


def smooth(y, k=7):
    """Centered moving average; identity for short series."""
    if len(y) < k:
        return np.asarray(y, dtype=float)
    w = np.ones(k) / k
    pad = k // 2
    yp = np.pad(y, (pad, pad), mode='edge')
    return np.convolve(yp, w, mode='valid')


def series(rows, key):
    xs, ys = [], []
    for r in rows:
        if key in r:
            xs.append(r['iter'])
            ys.append(r[key])
    return np.asarray(xs), np.asarray(ys, dtype=float)


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 8.5,
        'figure.dpi': 140,
        'savefig.dpi': 180,
        'axes.grid': True,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# Colourblind-friendly qualitative palette
C = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7',
     '#56B4E9', '#F0E442', '#000000']


def mark_ramp(ax):
    ax.axvline(RAMP_END, color='#888888', ls=':', lw=1.0, zorder=0)
    ymax = ax.get_ylim()[1]
    ax.text(RAMP_END, ymax, ' ramp end', color='#666666', fontsize=7.5,
            va='top', ha='left')


def plot_losses(train, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex=True)
    fig.suptitle('v5 training losses', fontweight='bold', y=0.98)

    # --- (a) stock mask losses, final layer ---
    ax = axes[0, 0]
    for i, (key, label) in enumerate([
            ('loss_cls', 'cls'),
            ('loss_mask', 'mask CE'),
            ('loss_dice', 'dice')]):
        x, y = series(train, key)
        ax.plot(x, smooth(y), color=C[i], lw=1.6, label=label)
        ax.plot(x, y, color=C[i], alpha=0.18, lw=0.6)
    ax.set_ylabel('loss')
    ax.set_title('(a) Stock losses (final decoder layer)')
    ax.legend(loc='upper right', frameon=False)

    # --- (b) inductive-bias aux losses (final layer; weighted as logged) ---
    ax = axes[0, 1]
    for i, (key, label) in enumerate([
            ('loss_boundary', 'BoundaryDice'),
            ('loss_nonoverlap', 'NonOverlap'),
            ('loss_surface', 'Surface (Kervadec)')]):
        x, y = series(train, key)
        ax.plot(x, smooth(y), color=C[i], lw=1.6, label=label)
        ax.plot(x, y, color=C[i], alpha=0.18, lw=0.6)
    mark_ramp(ax)
    ax.set_ylabel('weighted loss')
    ax.set_title('(b) Bias losses (final layer, as logged)')
    ax.legend(loc='upper right', frameon=False)
    ax.annotate(
        'NonOverlap / Surface ramped\n'
        'to full weight by iter 1650',
        xy=(0.55, 0.72), xycoords='axes fraction', fontsize=7.5,
        color='#555555', ha='left')

    # --- (c) dice across decoder layers ---
    ax = axes[1, 0]
    for i, (tag, prefix) in enumerate(LAYER_KEYS.items()):
        key = f'{prefix}.loss_dice' if prefix else 'loss_dice'
        label = f'{tag} dice' if tag != 'final' else 'final dice'
        x, y = series(train, key)
        ax.plot(x, smooth(y), color=C[i], lw=1.5, label=label)
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.set_title('(c) Dice across decoder layers')
    ax.legend(loc='upper right', frameon=False, ncol=2)

    # --- (d) surface across decoder layers ---
    ax = axes[1, 1]
    for i, (tag, prefix) in enumerate(LAYER_KEYS.items()):
        key = f'{prefix}.loss_surface' if prefix else 'loss_surface'
        label = f'{tag} surface' if tag != 'final' else 'final surface'
        x, y = series(train, key)
        ax.plot(x, smooth(y), color=C[i], lw=1.5, label=label)
    mark_ramp(ax)
    ax.set_xlabel('iteration')
    ax.set_ylabel('weighted loss')
    ax.set_title('(d) Surface loss across decoder layers')
    ax.legend(loc='lower left', frameon=False, ncol=2)

    for ax in axes.ravel():
        ax.set_xlim(0, MAX_ITERS)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out / 'v5_loss_curves.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')

    # Separate panel: raw (unit-weight) aux trajectories, undoing the ramp.
    # Logged value = raw * current_weight; invert so trends are comparable.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x_no, y_no = series(train, 'loss_nonoverlap')
    x_sf, y_sf = series(train, 'loss_surface')
    x_bd, y_bd = series(train, 'loss_boundary')

    def w_no(it):
        return 6.5 * min(it, RAMP_END) / RAMP_END if it > 0 else 1e-12

    def w_sf(it):
        return 0.001 + (0.09 - 0.001) * min(it, RAMP_END) / RAMP_END

    raw_no = np.array([y / max(w_no(it), 1e-9) for it, y in zip(x_no, y_no)])
    raw_sf = np.array([y / w_sf(it) for it, y in zip(x_sf, y_sf)])
    # BoundaryDice weight is fixed at 2.0
    raw_bd = y_bd / 2.0

    ax.plot(x_bd, smooth(raw_bd), color=C[0], lw=1.6, label='BoundaryDice (raw)')
    ax.plot(x_no, smooth(raw_no), color=C[1], lw=1.6, label='NonOverlap (raw)')
    ax.plot(x_sf, smooth(raw_sf), color=C[2], lw=1.6, label='Surface (raw)')
    mark_ramp(ax)
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss at unit weight')
    ax.set_title('v5 bias losses — weight-normalized (raw objective)')
    ax.set_xlim(0, MAX_ITERS)
    ax.legend(loc='upper right', frameon=False)
    fig.tight_layout()
    path = out / 'v5_loss_aux_raw.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


def plot_metrics(val, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex=True)
    fig.suptitle('v5 validation metrics', fontweight='bold', y=0.98)

    # --- (a) detection / region quality ---
    ax = axes[0, 0]
    for i, (key, label) in enumerate([
            ('coco/segm_mAP', 'segm mAP'),
            ('coco/segm_mAP_50', 'segm mAP@50'),
            ('coco/segm_mAP_75', 'segm mAP@75')]):
        x, y = series(val, key)
        ax.plot(x, y, color=C[i], lw=1.8, marker='o', ms=3.5, label=label)
    ax.set_ylabel('score')
    ax.set_title('(a) Region quality (COCO)')
    ax.legend(loc='lower right', frameon=False)
    ax.set_ylim(0, None)

    # --- (b) boundary adherence ---
    ax = axes[0, 1]
    for i, (key, label) in enumerate([
            ('Boundary-IoU', 'Boundary-IoU'),
            ('Boundary-F_1px', 'Boundary-F @1px'),
            ('Boundary-F_3px', 'Boundary-F @3px'),
            ('Boundary-F_5px', 'Boundary-F @5px')]):
        x, y = series(val, key)
        ax.plot(x, y, color=C[i], lw=1.8, marker='o', ms=3.5, label=label)
    ax.set_ylabel('score')
    ax.set_title('(b) Boundary adherence')
    ax.legend(loc='lower right', frameon=False)
    ax.set_ylim(0, 1.0)

    # --- (c) planar partition / topology ---
    ax = axes[1, 0]
    for i, (key, label) in enumerate([
            ('Over-segmentation_Rate', 'Over-seg rate'),
            ('Under-segmentation_Rate', 'Under-seg rate'),
            ('Duplicate_Rate', 'Duplicate rate')]):
        x, y = series(val, key)
        ax.plot(x, y, color=C[i], lw=1.8, marker='o', ms=3.5, label=label)
    x, y = series(val, 'Pred_GT_Count_Ratio')
    ax2 = ax.twinx()
    ax2.plot(x, y, color=C[4], lw=1.8, marker='s', ms=3.5,
             label='Pred/GT count')
    ax2.axhline(1.0, color=C[4], ls=':', lw=1.0, alpha=0.7)
    ax2.set_ylabel('count ratio', color=C[4])
    ax2.tick_params(axis='y', labelcolor=C[4])
    ax2.spines['right'].set_visible(True)
    ax.set_xlabel('iteration')
    ax.set_ylabel('rate')
    ax.set_title('(c) Planar partition (topology)')
    # merge legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', frameon=False)

    # --- (d) boundary regularity ---
    ax = axes[1, 1]
    x, y = series(val, 'Vertices_IoU95_pred')
    ax.plot(x, y, color=C[0], lw=1.8, marker='o', ms=3.5,
            label='Vertices@IoU0.95')
    ax.set_ylabel('vertices / instance', color=C[0])
    ax.tick_params(axis='y', labelcolor=C[0])
    ax2 = ax.twinx()
    x, y = series(val, 'Curvature_Energy_pred')
    ax2.plot(x, y, color=C[3], lw=1.8, marker='s', ms=3.5,
             label='Curvature energy')
    ax2.set_ylabel('curvature energy', color=C[3])
    ax2.tick_params(axis='y', labelcolor=C[3])
    ax2.spines['right'].set_visible(True)
    ax.set_xlabel('iteration')
    ax.set_title('(d) Boundary regularity')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', frameon=False)

    for ax in axes.ravel():
        ax.set_xlim(0, MAX_ITERS)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out / 'v5_eval_curves.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    train, val = load_log(LOG)
    print(f'parsed {len(train)} train points, {len(val)} val points')
    plot_losses(train, OUT)
    plot_metrics(val, OUT)


if __name__ == '__main__':
    main()
