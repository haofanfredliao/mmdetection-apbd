"""Find the largest train batch size that fits for a config on this GPU.

Does one real forward+backward with the train pipeline so activation
memory matches training. Reports peak allocated memory.

Usage:
    python scripts/probe_backbone_batch.py CONFIG [--start 12] [--min 2]
"""
import argparse
import gc

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from torch.utils.data import DataLoader

from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules


def try_bs(cfg_path, bs):
    register_all_modules()
    init_default_scope('mmdet')
    cfg = Config.fromfile(cfg_path)
    cfg.train_dataloader.batch_size = bs

    model = MODELS.build(cfg.model).cuda().train()
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    loader = DataLoader(
        dataset, batch_size=bs, shuffle=True, num_workers=0,
        collate_fn=pseudo_collate)
    raw = next(iter(loader))
    data = model.data_preprocessor(raw, True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    losses = model.loss(data['inputs'], data['data_samples'])
    total = sum(v for v in losses.values() if torch.is_tensor(v) and v.requires_grad)
    total.backward()
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

    del model, losses, total, data, raw, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--start', type=int, default=12)
    parser.add_argument('--min', type=int, default=2)
    args = parser.parse_args()

    print(f'probing {args.config}')
    bs = args.start
    while bs >= args.min:
        try:
            peak = try_bs(args.config, bs)
            print(f'  bs={bs:>2}  OK   peak={peak:.1f} GB')
            print(f'RECOMMEND batch_size={bs}')
            return
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise
            print(f'  bs={bs:>2}  OOM')
            torch.cuda.empty_cache()
            gc.collect()
            bs = bs // 2 if bs > 4 else bs - 1
    print('FAILED: even min batch size OOMs')


if __name__ == '__main__':
    main()
