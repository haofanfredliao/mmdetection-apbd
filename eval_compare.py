#!/usr/bin/env python3
"""
对两个 Mask2Former 模型（标准版 & 边界损失版）在 val 集上进行推理并对比指标。

推理结果 (pkl) 保存至 ~/data/ai4b_coco/predictions/
评估结果 (log/json) 保存至 ~/data/ai4b_coco/eval_results/
对比报告保存至              ~/data/ai4b_coco/eval_results/comparison_<时间戳>.txt

用法:
    python eval_compare.py              # 默认 CPU 推理（无 GPU 环境）
    python eval_compare.py --device cuda:0   # 使用 GPU 推理
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ===================================================================
# 模型配置
# ===================================================================
PROJECT_DIR = Path(__file__).parent.resolve()

MODELS = [
    {
        'name': 'mask2former_standard',
        'label': 'Mask2Former (Standard Dice Loss)',
        'config': 'configs/ai4boundary/mask2former_r50_1xb2-50e_custom.py',
        'checkpoint': 'work_dirs/mask2former_r50_1xb2-50e_custom/best_coco_segm_mAP_iter_16219.pth',
        'inject_field_metric': True,  # 原始配置缺少 FieldSegmentationMetric，自动注入
    },
    {
        'name': 'mask2former_boundary_v1',
        'label': 'Mask2Former (Boundary Dice Loss v1)',
        'config': 'configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v1.py',
        'checkpoint': 'work_dirs/mask2former_r50_1xb2-50e_custom_boundary_v1/best_coco_segm_mAP_iter_21312.pth',
    },
]

# 结果根目录（~/data/ai4b_coco/）
EVAL_BASE_DIR = Path.home() / 'data' / 'ai4b_coco' / 'eval_results'
PRED_BASE_DIR = Path.home() / 'data' / 'ai4b_coco' / 'predictions'
VIS_BASE_DIR  = Path.home() / 'data' / 'ai4b_coco' / 'vis_results'

# 需要优先展示的指标（按此顺序）
# 注：FieldSegmentationMetric 实际输出 Over/Under-segmentation_Rate
PRIORITY_METRICS = [
    'coco/segm_mAP',
    'coco/segm_mAP_50',
    'coco/segm_mAP_75',
    'coco/segm_mAP_s',
    'coco/segm_mAP_m',
    'coco/segm_mAP_l',
    'coco/bbox_mAP',
    'coco/bbox_mAP_50',
    'coco/bbox_mAP_75',
    'Over-segmentation_Rate',
    'Under-segmentation_Rate',
]


# ===================================================================
# 推理 & 评估
# ===================================================================

def _make_field_metric_config(base_config: str, data_root: str, tmp_dir: Path) -> str:
    """
    为缺少 FieldSegmentationMetric 的模型生成一个临时的 override 配置文件。
    返回临时配置文件路径。
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg = tmp_dir / 'field_metric_override.py'
    ann_file = f'{data_root}annotations/instances_val.json'
    tmp_cfg.write_text(
        f"_base_ = ['{base_config}']\n"
        f"val_evaluator = [\n"
        f"    dict(\n"
        f"        type='CocoMetric',\n"
        f"        ann_file='{ann_file}',\n"
        f"        metric=['bbox', 'segm'],\n"
        f"        format_only=False),\n"
        f"    dict(type='FieldSegmentationMetric', iou_thr=0.5),\n"
        f"]\n"
    )
    return str(tmp_cfg)


def run_test(model_cfg: dict, device: str) -> dict:
    """调用 tools/test.py 评估单个模型，返回解析的指标字典。"""
    work_dir = EVAL_BASE_DIR / model_cfg['name']
    work_dir.mkdir(parents=True, exist_ok=True)

    out_file = PRED_BASE_DIR / f"{model_cfg['name']}_val_pred.pkl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 可视化图片保存目录（--show-dir 是相对 work_dir 的子路径名）
    vis_subdir = str(VIS_BASE_DIR / model_cfg['name'])

    # 若模型配置缺少 FieldSegmentationMetric，生成临时 override 配置
    config_path = model_cfg['config']
    if model_cfg.get('inject_field_metric'):
        data_root = 'data/data/ai4b_coco/'
        config_path = _make_field_metric_config(
            base_config=str(PROJECT_DIR / model_cfg['config']),
            data_root=data_root,
            tmp_dir=work_dir / '_tmp_cfg',
        )
        print(f'[INFO] 已生成临时配置（注入 FieldSegmentationMetric）: {config_path}')

    cmd = [
        sys.executable, 'tools/test.py',
        config_path,
        model_cfg['checkpoint'],
        '--work-dir', str(work_dir),
        '--out', str(out_file),
        '--show-dir', vis_subdir,
    ]

    env = os.environ.copy()
    # CPU 模式：屏蔽所有 GPU，让 mmengine 自动回退到 CPU
    if device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    elif device.startswith('cuda'):
        # 提取设备编号，如 cuda:0 → 0
        gpu_id = device.split(':')[-1] if ':' in device else '0'
        env['CUDA_VISIBLE_DEVICES'] = gpu_id

    print(f"\n{'='*72}")
    print(f"  模型  : {model_cfg['label']}")
    print(f"  配置  : {model_cfg['config']}")
    print(f"  权重  : {model_cfg['checkpoint']}")
    print(f"  设备  : {device}")
    print(f"  结果目录 : {work_dir}")
    print(f"  预测文件 : {out_file}")
    print(f"  可视化图片: {work_dir}/<时间戳>/{vis_subdir}/")
    print(f"{'='*72}\n")

    # 实时流输出，同时捕获所有行用于解析指标
    captured_lines: list = []
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(PROJECT_DIR),
        env=env,
    )
    for line in process.stdout:
        print(line, end='', flush=True)
        captured_lines.append(line)
    process.wait()

    if process.returncode != 0:
        print(f'\n[WARNING] tools/test.py 以非零退出码结束: {process.returncode}')

    # 优先从 stdout 最后的 Iter(test) 行解析指标（最可靠）
    metrics = parse_metrics_from_stdout(captured_lines)
    if not metrics:
        # 备用：尝试读 scalars.json
        metrics = parse_scalars_json(work_dir)
    return metrics


# 匹配 mmengine 测试结束行，例如：
# Iter(test) [285/285]    coco/bbox_mAP: 0.3670  coco/segm_mAP: 0.3530 ...
_ITER_TEST_RE = re.compile(r'Iter\(test\)\s+\[\d+/\d+\](.+)')
# 匹配单个 key: value 对（value 为浮点数）
_KV_RE = re.compile(r'([\w/\-]+):\s+([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)')


def parse_metrics_from_stdout(lines: list) -> dict:
    """
    从进程 stdout 捕获行中解析最终的测试指标。
    mmengine 在 test 结束时会输出：
      Iter(test) [N/N]  metric1: val1  metric2: val2 ...
    排除 data_time / time / memory 等非指标字段。
    """
    SKIP_KEYS = {'data_time', 'time', 'memory', 'iter', 'step'}
    metrics = {}
    for line in reversed(lines):
        m = _ITER_TEST_RE.search(line)
        if m:
            for key, val in _KV_RE.findall(m.group(1)):
                if key not in SKIP_KEYS:
                    metrics[key] = float(val)
            if metrics:
                print(f'[INFO] 从 stdout 解析到 {len(metrics)} 个指标')
                return metrics
    return {}


def parse_scalars_json(work_dir: Path) -> dict:
    """
    备用：从 work_dir/{timestamp}/vis_data/scalars.json 解析评估指标。
    """
    candidates = sorted(work_dir.glob('*/vis_data/scalars.json'))
    if not candidates:
        print(f'[WARNING] 未找到 scalars.json，目录: {work_dir}')
        return {}

    scalars_path = candidates[-1]
    print(f'[INFO] 读取指标文件: {scalars_path}')

    metrics = {}
    try:
        with open(scalars_path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for line in reversed(lines):
            data = json.loads(line)
            if any(k.startswith(('coco/', 'field_seg/')) or k in ('Over-segmentation_Rate', 'Under-segmentation_Rate') for k in data):
                metrics = {
                    k: v for k, v in data.items()
                    if isinstance(v, (int, float))
                    and k not in ('data_time', 'time', 'memory', 'iter', 'step')
                }
                break
    except Exception as exc:
        print(f'[WARNING] 解析 scalars.json 失败: {exc}')

    return metrics


# ===================================================================
# 结果展示
# ===================================================================

def print_comparison(all_results: dict):
    """打印对比表并将报告保存至文件。"""
    model_names = list(all_results.keys())
    labels = [m['label'] for m in MODELS if m['name'] in model_names]

    # 收集所有指标键（保持出现顺序，去重）
    seen: set = set()
    all_keys: list = []
    for metrics in all_results.values():
        for k in metrics:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    # 优先指标排前，剩余按字母序补充
    sorted_keys = [k for k in PRIORITY_METRICS if k in seen]
    sorted_keys += sorted(k for k in all_keys if k not in PRIORITY_METRICS)

    if not sorted_keys:
        print('[WARNING] 没有可对比的数值指标，请检查输出日志。')
        return

    key_w = max(len(k) for k in sorted_keys) + 2
    col_w = max(24, max(len(lb) for lb in labels) + 4)
    sep = '=' * (key_w + col_w * len(labels) + 2)

    lines = [
        '',
        sep,
        f'  EVALUATION COMPARISON  —  val split  —  {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        sep,
        f"{'Metric':<{key_w}}" + ''.join(f'{lb:>{col_w}}' for lb in labels),
        '-' * (key_w + col_w * len(labels)),
    ]

    for key in sorted_keys:
        vals = [all_results[n].get(key) for n in model_names]
        numeric_vals = [v for v in vals if isinstance(v, (int, float))]
        best = max(numeric_vals) if numeric_vals else None

        row = f'{key:<{key_w}}'
        for v in vals:
            if isinstance(v, float):
                cell = f'{v:.4f}{"  ◀" if v == best and len(numeric_vals) > 1 else "   "}'
            elif isinstance(v, int):
                cell = f'{v}{"  ◀" if v == best and len(numeric_vals) > 1 else ""}'
            else:
                cell = 'N/A'
            row += f'{cell:>{col_w}}'
        lines.append(row)

    lines += [sep, '  ◀ 表示该指标最高值', '']

    output = '\n'.join(lines)
    print(output)

    # 保存报告
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = EVAL_BASE_DIR / f'comparison_{ts}.txt'
    EVAL_BASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(output)
        f.write('\n\nFull metrics (JSON):\n')
        f.write(json.dumps(all_results, indent=2))
    print(f'[INFO] 对比报告已保存至: {report_path}')


# ===================================================================
# 主入口
# ===================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate and compare two Mask2Former models on val set')
    parser.add_argument(
        '--device', default='cpu',
        help='推理设备，如 cpu / cuda:0 （默认: cpu，无需 GPU 资源）')
    parser.add_argument(
        '--model', choices=[m['name'] for m in MODELS] + ['all'], default='all',
        help='只评估指定模型，默认评估全部')
    return parser.parse_args()


def main():
    args = parse_args()

    # 校验 checkpoint 是否存在
    for m in MODELS:
        ckpt = PROJECT_DIR / m['checkpoint']
        if not ckpt.exists():
            print(f'[ERROR] 权重文件不存在: {ckpt}')
            print('  请确认训练已完成，或手动修改脚本中的 checkpoint 路径。')
            sys.exit(1)

    models_to_run = MODELS if args.model == 'all' else [m for m in MODELS if m['name'] == args.model]

    all_results: dict = {}
    for model_cfg in models_to_run:
        metrics = run_test(model_cfg, device=args.device)
        all_results[model_cfg['name']] = metrics

        if metrics:
            print(f'\n[INFO] {model_cfg["name"]} 关键指标:')
            for k in PRIORITY_METRICS[:6]:
                if k in metrics:
                    print(f'  {k}: {metrics[k]:.4f}')

    # 保存完整指标 JSON
    json_path = EVAL_BASE_DIR / 'all_metrics.json'
    EVAL_BASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n[INFO] 完整指标 JSON 已保存至: {json_path}')

    # 打印对比表
    if len(all_results) >= 2:
        print_comparison(all_results)
    elif len(all_results) == 1:
        name = list(all_results.keys())[0]
        print(f'\n[INFO] 仅评估了一个模型 ({name})，跳过对比表。')


if __name__ == '__main__':
    main()
