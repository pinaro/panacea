#!/bin/sh
#SBATCH -J evalPolicy
#SBATCH -n 1
#SBATCH --partition=longq
#SBATCH --time=10-01:00:00

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

srun python3 /home/pinar/Safe-Secure-RL/SimGlucose/create_dataset.py
