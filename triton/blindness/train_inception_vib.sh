#!/bin/bash
#SBATCH --time=32:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --output=train_inception_vib.out

module load cuda
module load anaconda
srun --gres=gpu:1 python3 train_inception_vib.py
