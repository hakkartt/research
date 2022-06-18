#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --output=train_resnet_nonweighted.out

module load cuda
module load anaconda
srun --gres=gpu:1 python3 train_resnet_nonweighted.py
