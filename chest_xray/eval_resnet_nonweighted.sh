#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=500M
#SBATCH --output=eval_resnet_nonweighted.out

module load anaconda
srun python3 eval_resnet_nonweighted.py
