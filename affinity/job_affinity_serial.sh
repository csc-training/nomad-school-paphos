#!/bin/bash -l
#SBATCH --job-name=affinity
#SBATCH --account=project_465000539
#SBATCH --partition=small
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:05:00       

srun ./check_affinity

