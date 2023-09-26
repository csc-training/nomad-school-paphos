#!/bin/bash -l
#SBATCH --job-name=affinity
#SBATCH --account=project_465000539
#SBATCH --partition=debug
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
#SBATCH --time=0:05:00       

srun ./check_affinity

