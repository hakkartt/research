#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --output=eval_resnet_vib_cw.out

module load cuda
module load anaconda
srun --gres=gpu:1 python3 eval_resnet_vib_cw.py
