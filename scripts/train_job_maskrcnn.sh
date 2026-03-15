#!/bin/bash
#SBATCH --job-name=mask_rcnn_ai4b
#SBATCH --partition=GEOG-HPC-GPU     # 指定 GPU 分区 (参考手册 3.1)
#SBATCH --qos=Normal                 # 指定 QoS (参考手册 3.2)
#SBATCH --nodes=1                    # 使用 1 个节点
#SBATCH --ntasks=1                   # 运行 1 个任务
#SBATCH --cpus-per-task=8            # 为数据加载分配 8 个 CPU 核心 (对应 dataloader 的 workers)
#SBATCH --mem=32G                    # 申请 32GB 内存
#SBATCH --gres=shard:1                 # 申请 1 块 80GB GPU (参考手册 3.5)
#SBATCH --time=12:00:00            # 最大运行时间 12小时 (格式 D-HH:MM:SS)
#SBATCH --output=logs/train_%j.out   # 标准输出日志保存路径 (%j 会被替换为真实的 Job ID)
#SBATCH --error=logs/train_%j.err    # 错误日志保存路径

source ~/miniconda3/etc/profile.d/conda.sh  # 加载 conda 环境
conda activate mmdet-py38  # 激活 conda 环境

cd $HOME/code/mmdetection-apbd

python tools/train.py configs/ai4boundary/mask-rcnn_r50_fpn_1x_ai4b.py