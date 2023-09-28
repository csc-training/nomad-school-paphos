# Simple GPU usage

In this exercise you can have a first taste about using GPUs.

1. Use the provided batch job script [job_gpu.sh](job_gpu.sh) for running
   the `rocm-smi` command in GPU node. Investigate output.

   Submit the batch job again, but request now more GPUs, *i.e.* increase
   the value in `--gpus-per-node` Slurm option. How does the output look now?

   Hint: `rocm-smi` can be useful also with real applications. While a GPU job is 
   running, one can [start a shell in compute node](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/interactive/#using-srun-to-check-running-jobs), and by issuing
   the `rocm-smi` command there one can get first hint how the application is 
   utilizing GPUs.

2. Try to build and run a simple "GPU hello world" program. In order to build application
   for GPUs, a proper modules need to be loaded:

   ```
   module load PrgEnv-cray
   module load rocm
   module load craype-accel-amd-gfx90a
   ```

   Once the modules are loaded, build the code with the provided [Makefile](Makefile).
   Finally, run the program `hello_gpu` in a GPU node, try to request different number of
   GPUs.


