#!/usr/bin/env python
# ============================================================
# 用训练好的 checkpoint 对几张测试图跑推理，画出预测的 mask。
#
# 按需求调整了可视化：
#   - 不画 bbox 矩形框（DetLocalVisualizer(draw_bbox=False)）
#   - mask 的白色轮廓线调窄（mask_edge_width=1，默认是 2）
#
# 用法：
#   python scripts/visualize_inference.py \
#       --config configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v4_combined.py \
#       --checkpoint work_dirs/.../best_coco_segm_mAP_iter_13860.pth \
#       --images AT_10038 AT_10260 AT_3801 AT_4079 \
#       --out-dir outputs/v4_combined_demo
# ============================================================
import argparse
import glob
import os.path as osp

import mmcv
from mmengine.utils import mkdir_or_exist

from mmdet.apis import inference_detector, init_detector
from mmdet.visualization import DetLocalVisualizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument(
        '--image-dir', default='data/data/ai4b_coco/images/test')
    parser.add_argument(
        '--images',
        nargs='+',
        required=True,
        help='sample_id list (no extension), e.g. AT_10038')
    parser.add_argument(
        '--ext',
        default='.png',
        help='Used as a glob suffix: <image-dir>/<sample_id>*<ext>, since '
        'AI4Boundary filenames have a suffix after the sample id '
        '(e.g. AT_10038_ortho_1m_512.png).')
    parser.add_argument('--out-dir', default='outputs/inference_demo')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.0,
        help='The model already applies its own score_thr during '
        'postprocessing (see test_cfg in the config); keep this at 0.0 '
        'so the visualizer does not double-filter.')
    parser.add_argument('--mask-edge-width', type=float, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    mkdir_or_exist(args.out_dir)

    model = init_detector(args.config, args.checkpoint, device=args.device)

    visualizer = DetLocalVisualizer(
        name='inference_demo',
        draw_bbox=False,
        mask_edge_width=args.mask_edge_width,
        alpha=0.6)
    visualizer.dataset_meta = model.dataset_meta

    for sample_id in args.images:
        matches = sorted(
            glob.glob(osp.join(args.image_dir, sample_id + '*' + args.ext)))
        if not matches:
            print(f'!!! no image found for {sample_id}, skipping')
            continue
        img_path = matches[0]
        result = inference_detector(model, img_path)
        img = mmcv.imread(img_path, channel_order='rgb')

        out_file = osp.join(args.out_dir, f'{sample_id}_pred.png')
        visualizer.add_datasample(
            sample_id,
            img,
            data_sample=result,
            draw_gt=False,
            draw_pred=True,
            pred_score_thr=args.pred_score_thr,
            out_file=out_file)
        n_inst = len(result.pred_instances)
        print(f'{sample_id}: {n_inst} instances -> {out_file}')


if __name__ == '__main__':
    main()
