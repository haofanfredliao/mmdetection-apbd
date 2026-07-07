#!/bin/bash
#SBATCH --job-name=eval_e1b_tune
#SBATCH --partition=GEOG-HPC-GPU
#SBATCH --qos=Normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=shard:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_e1b_%j.out
#SBATCH --error=logs/eval_e1b_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdet-py38

cd $HOME/code/mmdetection-apbd

python eval_e1b_postprocess_tune.py --device cuda:0
