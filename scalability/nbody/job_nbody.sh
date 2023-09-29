#!/bin/bash -l
#SBATCH --job-name=nbody
#SBATCH --account=project_465000539
#SBATCH --partition=small
#SBATCH --nodes=2               
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --time=0:15:00       

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores

srun ./nbody 25600 1000

