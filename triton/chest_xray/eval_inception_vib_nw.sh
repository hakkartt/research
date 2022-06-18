#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=eval_inception_vib_nw.out

module load cuda
module load anaconda
srun --gres=gpu:1 python3 eval_inception_vib_nw.py
