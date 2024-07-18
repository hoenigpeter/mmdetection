#!/bin/bash
#SBATCH --job-name=yolox_x_keypose_mix
#SBATCH --partition=GPU-v100
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --time=120:00:00
#SBATCH --output=slurm_yolox_x_keypose_mix
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=hoenig@acin.tuwien.ac.at
#SBATCH --mail-type=BEGIN,END,FAIL

cd mmdetection && python tools/train.py configs/yolox/yolox_x_8xb8-300e_keypose_mix.py --auto-scale-lr
