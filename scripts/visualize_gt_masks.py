#!/usr/bin/env python
"""Visualize GT masks in the same style as visualize_inference.py.

For each sample_id: copy/save the original RGB image, and overlay GT instance
masks with DetLocalVisualizer (no bbox, thin white edges, alpha=0.6).

Usage:
  python scripts/visualize_gt_masks.py \
      --images ES_703 NL_2434 ... \
      --out-dir outputs/gt_reference
"""
import argparse
import glob
import os.path as osp
import shutil

import mmcv
import numpy as np
from mmengine.utils import mkdir_or_exist
from pycocotools import mask as mask_util
from pycocotools.coco import COCO

from mmdet.structures.mask import bitmap_to_polygon
from mmdet.visualization import DetLocalVisualizer
from mmdet.visualization.palette import get_palette, jitter_color


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann-file',
                        default='data/data/ai4b_coco/annotations/instances_test.json')
    parser.add_argument('--image-dir', default='data/data/ai4b_coco/images/test')
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--ext', default='.png')
    parser.add_argument('--out-dir', default='outputs/gt_reference')
    parser.add_argument('--mask-edge-width', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=0.6)
    return parser.parse_args()


def ann_to_bitmap(seg, h, w):
    if isinstance(seg, list):
        rles = mask_util.frPyObjects(seg, h, w)
        rle = mask_util.merge(rles)
    elif isinstance(seg, dict) and isinstance(seg.get('counts'), list):
        rle = mask_util.frPyObjects(seg, h, w)
    else:
        rle = seg
    return mask_util.decode(rle).astype(bool)


def main():
    args = parse_args()
    mkdir_or_exist(args.out_dir)

    coco = COCO(args.ann_file)
    # file_name stem -> image info
    by_stem = {}
    for img in coco.dataset['images']:
        stem = osp.splitext(img['file_name'])[0]
        # AI4Boundary: ES_703_ortho_1m_512 -> sample id ES_703
        sample_id = stem.split('_ortho')[0] if '_ortho' in stem else stem
        by_stem[sample_id] = img

    visualizer = DetLocalVisualizer(
        name='gt_reference',
        draw_bbox=False,
        mask_edge_width=args.mask_edge_width,
        alpha=args.alpha)
    visualizer.dataset_meta = dict(
        classes=('field', ),
        palette=[(220, 20, 60)])

    for sample_id in args.images:
        matches = sorted(
            glob.glob(osp.join(args.image_dir, sample_id + '*' + args.ext)))
        if not matches:
            print(f'!!! no image for {sample_id}, skipping')
            continue
        img_path = matches[0]
        img_info = by_stem.get(sample_id)
        if img_info is None:
            # fallback: match by file_name prefix
            base = osp.basename(img_path)
            hits = [im for im in coco.dataset['images']
                    if im['file_name'] == base or im['file_name'].startswith(sample_id)]
            if not hits:
                print(f'!!! no annotation for {sample_id}, skipping')
                continue
            img_info = hits[0]

        # --- original image ---
        raw_out = osp.join(args.out_dir, f'{sample_id}_image.png')
        shutil.copy2(img_path, raw_out)

        img = mmcv.imread(img_path, channel_order='rgb')
        h, w = img.shape[:2]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_info['id']))

        masks = []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            m = ann_to_bitmap(ann['segmentation'], h, w)
            if m.sum() == 0:
                continue
            masks.append(m)

        # Draw masks only — no class/score text labels.
        visualizer.set_image(img)
        if masks:
            mask_arr = np.stack(masks, 0).astype(bool)
            palette = get_palette([(220, 20, 60)], 1)
            colors = [jitter_color(palette[0]) for _ in masks]
            polygons = []
            for m in mask_arr:
                contours, _ = bitmap_to_polygon(m)
                polygons.extend(contours)
            visualizer.draw_polygons(
                polygons, edge_colors='w', alpha=args.alpha,
                line_widths=args.mask_edge_width)
            visualizer.draw_binary_masks(
                mask_arr, colors=colors, alphas=args.alpha)

        mask_out = osp.join(args.out_dir, f'{sample_id}_gt.png')
        mmcv.imwrite(mmcv.rgb2bgr(visualizer.get_image()), mask_out)
        print(f'{sample_id}: {len(masks)} GT -> {raw_out} , {mask_out}')


if __name__ == '__main__':
    main()
