# Using LUMI

More detailed instructions are provided in the [LUMI user guide](https://docs.lumi-supercomputer.eu), here are brief instructions for basic usage

## Getting access

You should have received an email with invitation to the LUMI project. In order to 
proceed with the invitation, one needs to first create MyAccessID, and once that is 
completed, click the "accept invitation” link in the email. 

## Connecting to LUMI

LUMI can be accessed only with ssh keys. The public key needs to be registered to
your MyAccessID user profile. From there, the public key will be copied to LUMI.
Note that there can be a couple of hours delay until the public key is synchronized to 
LUMI.

When ssh key setup is complete, you can login with
```
ssh -i <path-to-private-key> <username>@lumi.csc.fi
```

## Disk areas

It is recommended that all the hands-on exercises in LUMI are carried out in the
**scratch** disk area. The name of the scratch directory can be
queried with the command `lumi-workspaces`. As the base directory is
shared between members of the project, you should create your own
directory:
```
cd /scratch/project_465000539/
mkdir -p $USER
cd $USER
```

## Module system

LUMI has several different programming environments available via the module system.

Some application codes and tools used in the school are also available via modules, 
in order to use those you need to first issue the command
```
module use /scratch/project_465000539/modules
```

After that, e.g. **abinit** is available for use with
```
module load abinit
```

## Batch jobs

Programs need to be executed in the compute nodeds via the batch job system. A simple
example for running **abinit** with 4 MPI tasks and with 4 OpenMP threads 
(using 16 CPU cores in total):

```
#!/bin/bash
#SBATCH --job-name=abinit_example
#SBATCH --account=project_465000539
#SBATCH --partition=debug
#SBATCH --time=00:25:00
#SBATCH --nodes=1   # each node has 128 cores
#SBATCH --ntasks-per-node=4  # reserve 4 cores per node for MPI
#SBATCH --cpus-per-task=4    # reserve 4 cores for each MPI task for threading
#SBATCH --mem-per-cpu=2G     # reserve 2 GiB of memory for each core

odule use /scratch/project_465000539/modules
module load abinit

# OpenMp Environment
# use OpenMP threads based on --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun abinit tbase1_1.abi > log 2> err

```

Save the script *e.g.* as `abinit_job.sh` and submit it with `sbatch job.sh`.
The output of job will be in file `slurm-xxxxx.out`. You can check the status of your jobs with `squeue --me` and kill possible hanging applications with
`scancel JOBID`.

