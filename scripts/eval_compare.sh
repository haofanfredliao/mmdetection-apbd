#!/bin/bash
#SBATCH --job-name=eval_compare
#SBATCH --partition=GEOG-HPC-GPU     
#SBATCH --qos=Normal                 # 指定 QoS 
#SBATCH --nodes=1                    # 使用 1 个节点
#SBATCH --ntasks=1                   # 运行 1 个任务
#SBATCH --cpus-per-task=8            # 为数据加载分配 8 个 CPU 核心
#SBATCH --mem=64G                    # 申请 64GB 内存
#SBATCH --time=1-00:00:00            # 最大运行时间 1天
#SBATCH --output=logs/eval_%j.out    # 标准输出日志保存路径
#SBATCH --error=logs/eval_%j.err     # 错误日志保存路径

source ~/miniconda3/etc/profile.d/conda.sh  
conda activate mmdet-py38  

cd $HOME/code/mmdetection-apbd

python scripts/infer_eval_compare.py
