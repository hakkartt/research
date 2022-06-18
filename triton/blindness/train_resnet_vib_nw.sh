#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --output=train_resnet_vib_nw.out

module load cuda
module load anaconda
srun --gres=gpu:1 python3 train_resnet_vib_nw.py
