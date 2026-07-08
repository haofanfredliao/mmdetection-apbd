#!/bin/bash
#SBATCH --job-name=m2f_bound_v3_ov
#SBATCH --partition=GEOG-HPC-GPU
#SBATCH --qos=Normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=shard:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_v3_ov_%j.out
#SBATCH --error=logs/train_v3_ov_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdet-py38

cd $HOME/code/mmdetection-apbd

python tools/train.py configs/ai4boundary/mask2former_r50_1xb2-50e_custom_boundary_v3_nonoverlap_shard.py
