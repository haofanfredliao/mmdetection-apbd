#!/usr/bin/env python
# ============================================================
# 在 visualize_inference.py 基础上，额外可视化 backbone 抽取的多尺度
# 特征图（ResNet 的 4 个 stage，对应 stride 4/8/16/32）。
#
# 做法：给 model.backbone 挂一个 forward hook，在 inference_detector
# 触发的前向传播过程中把 4 个 stage 的输出张量截下来，然后用
# mmengine Visualizer.draw_featmap(channel_reduction='squeeze_mean')
# 把每个 stage 的多通道特征压成一张热力图，叠加在原图上。
#
# 每张输入图片最终生成一张拼接大图（预测结果 + 4 个 stage 热力图，
# 一行五列），方便一次性对比查看。
#
# 用法：
#   python scripts/visualize_inference_featmap.py \
#       --config configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v4_combined.py \
#       --checkpoint work_dirs/.../best_coco_segm_mAP_iter_13860.pth \
#       --images ES_703 NL_2434 ... \
#       --out-dir outputs/v4_combined_featmap
# ============================================================
import argparse
import glob
import os.path as osp

import matplotlib.pyplot as plt
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
    parser.add_argument('--out-dir', default='outputs/inference_featmap_demo')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.0,
        help='The model already applies its own score_thr during '
        'postprocessing (see test_cfg in the config); keep this at 0.0 '
        'so the visualizer does not double-filter.')
    parser.add_argument('--mask-edge-width', type=float, default=1)
    parser.add_argument(
        '--channel-reduction',
        default='squeeze_mean',
        choices=['squeeze_mean', 'select_max'],
        help='How draw_featmap collapses the C channels of each backbone '
        'stage down to a single-channel heatmap.')
    return parser.parse_args()


def main():
    args = parse_args()
    mkdir_or_exist(args.out_dir)

    model = init_detector(args.config, args.checkpoint, device=args.device)

    # backbone(x) 对 ResNet 直接返回 (feat_stage1, ..., feat_stage4) 这个
    # tuple，用 forward hook 把它截下来即可，不用重复 data_preprocessor
    # 的预处理逻辑（保证特征图和展示的预测结果对应同一份预处理后的输入）。
    captured = {}

    def _hook(_module, _inputs, outputs):
        captured['feats'] = outputs

    handle = model.backbone.register_forward_hook(_hook)

    visualizer = DetLocalVisualizer(
        name='inference_featmap_demo',
        draw_bbox=False,
        mask_edge_width=args.mask_edge_width,
        alpha=0.6)
    visualizer.dataset_meta = model.dataset_meta

    strides = [4, 8, 16, 32]

    for sample_id in args.images:
        matches = sorted(
            glob.glob(osp.join(args.image_dir, sample_id + '*' + args.ext)))
        if not matches:
            print(f'!!! no image found for {sample_id}, skipping')
            continue
        img_path = matches[0]
        img = mmcv.imread(img_path, channel_order='rgb')

        result = inference_detector(model, img_path)
        feats = captured['feats']
        n_stage = len(feats)

        # ---- 预测结果图（复用 visualize_inference.py 的画法）----
        pred_file = osp.join(args.out_dir, f'{sample_id}_pred.png')
        visualizer.add_datasample(
            sample_id,
            img,
            data_sample=result,
            draw_gt=False,
            draw_pred=True,
            pred_score_thr=args.pred_score_thr,
            out_file=pred_file)
        pred_img = mmcv.imread(pred_file, channel_order='rgb')
        n_inst = len(result.pred_instances)

        # ---- 每个 stage 的特征热力图 ----
        featmap_imgs = []
        for i, feat in enumerate(feats):
            heatmap = visualizer.draw_featmap(
                feat[0],  # (C, H, W)，去掉 batch 维
                overlaid_image=img,
                channel_reduction=args.channel_reduction)
            featmap_imgs.append(heatmap)
            single_file = osp.join(
                args.out_dir, f'{sample_id}_feat_stage{i + 1}.png')
            mmcv.imwrite(mmcv.rgb2bgr(heatmap), single_file)

        # ---- 拼成一行：预测结果 + 4 个 stage 热力图 ----
        fig, axes = plt.subplots(1, n_stage + 1, figsize=(4 * (n_stage + 1), 4))
        axes[0].imshow(pred_img)
        axes[0].set_title(f'{sample_id}\npred ({n_inst} inst)')
        axes[0].axis('off')
        for i, heatmap in enumerate(featmap_imgs):
            axes[i + 1].imshow(heatmap)
            axes[i + 1].set_title(f'backbone stage{i + 1} (stride {strides[i]})')
            axes[i + 1].axis('off')
        fig.tight_layout()
        panel_file = osp.join(args.out_dir, f'{sample_id}_panel.png')
        fig.savefig(panel_file, dpi=120)
        plt.close(fig)

        print(f'{sample_id}: {n_inst} instances -> {panel_file}')

    handle.remove()


if __name__ == '__main__':
    main()
