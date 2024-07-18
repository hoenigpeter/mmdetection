#!/bin/bash
#SBATCH --job-name=frcnn_keypose_random_texture
#SBATCH --partition=GPU-v100
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --time=120:00:00
#SBATCH --output=slurm_frcnn_keypose_random_texture
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=hoenig@acin.tuwien.ac.at
#SBATCH --mail-type=BEGIN,END,FAIL

python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_2x_keypose_random_texture.py --auto-scale-lr
