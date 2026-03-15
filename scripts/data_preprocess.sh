#!/bin/bash
#SBATCH --job-name=data_prep
#SBATCH --partition=GEOG-HPC-GPU
#SBATCH --qos=Normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       
#SBATCH --mem=32G               
#SBATCH --time=02:00:00        

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdet-py38
cd $HOME/code/mmdetection-apbd
python convert_to_coco.py