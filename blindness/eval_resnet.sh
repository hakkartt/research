#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=500M
#SBATCH --output=eval_resnet.out

module load anaconda
srun python3 eval_resnet.py
