#!/bin/bash
#SBATCH --job-name=m2f_bound_v2
#SBATCH --partition=GEOG-HPC-GPU     # 指定 GPU 分区
#SBATCH --qos=Normal                 # 指定 QoS
#SBATCH --nodes=1                    # 使用 1 个节点
#SBATCH --ntasks=1                   # 运行 1 个任务
#SBATCH --cpus-per-task=8            # 为数据加载分配 8 个 CPU 核心
#SBATCH --mem=64G                    # 申请 64GB 内存
#SBATCH --gres=gpu:1                 # 申请 1 块 GPU
#SBATCH --time=1-00:00:00            # 最大运行时间 1天 (格式 D-HH:MM:SS)
#SBATCH --output=logs/train_%j.out   # 标准输出日志保存路径
#SBATCH --error=logs/train_%j.err    # 错误日志保存路径

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdet-py38

cd $HOME/code/mmdetection-apbd

python tools/train.py configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v2.py
