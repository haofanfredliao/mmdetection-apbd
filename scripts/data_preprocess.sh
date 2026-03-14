#!/bin/bash
#SBATCH --job-name=data_prep
#SBATCH --partition=GEOG-HPC-GPU
#SBATCH --qos=Normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # 预处理通常是多线程的，多申请一些 CPU (比如 16 核)
#SBATCH --mem=128G               # 申请大内存处理大型遥感数据
# 注意：这里去掉了 --gres=gpu:1，表示不使用 GPU
#SBATCH --time=1-00:00:00        # 预估 1 天内跑完

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdet-py38
cd /home/$USER/code/mmdetection-apbd
python convert_to_coco.py