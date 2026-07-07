#!/bin/bash
#SBATCH --job-name=smoke_v3_ov
#SBATCH --partition=GEOG-HPC-GPU
#SBATCH --qos=Normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=shard:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_v3_ov_%j.out
#SBATCH --error=logs/smoke_v3_ov_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdet-py38

cd $HOME/code/mmdetection-apbd

python tools/train.py \
  configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v3_nonoverlap.py \
  --work-dir work_dirs/smoke_mask2former_boundary_v3_nonoverlap \
  --cfg-options \
    train_dataloader.batch_size=2 \
    train_dataloader.num_workers=4 \
    train_cfg.max_iters=20 \
    train_cfg.val_interval=1000 \
    default_hooks.checkpoint.interval=1000
