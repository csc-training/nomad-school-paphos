#!/bin/bash -l
#SBATCH --job-name=gpu-example
#SBATCH --account=project_465000539
#SBATCH --partition=dev-g
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=0:05:00       

srun ./heat_openmp
