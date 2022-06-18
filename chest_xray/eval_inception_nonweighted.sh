#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=500M
#SBATCH --output=eval_inception_nonweighted.out

module load anaconda
srun python3 eval_inception_nonweighted.py
