#!/bin/bash -l
#SBATCH --job-name=heat
#SBATCH --account=project_465000539
#SBATCH --partition=small
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0:05:00       


# export OMP_NUM_THREADS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export OMP_PROC_BIND=spread
# export OMP_PLACES=cores

# export OMP_AFFINITY_FORMAT="Process %P level %L thread %0.3n affinity %A"
# export OMP_DISPLAY_AFFINITY=true

srun ./heat

