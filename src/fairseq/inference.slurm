#!/bin/sh
#SBATCH --job-name inference
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=16GB
#SBATCH --time=120:30:00

# ==== Main ======

cd fairseq
python final-ebr.py -method ebr -dataset iwdeen

