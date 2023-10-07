# Performance aspects when using multiple GPUs

In this exercise we examine some important performance aspects of
MPI programs that use multiple GPUs. As an example, we use the three dimensional
heat equation solver.

1. Build the basic version with the provided [Makefile](heat-equation/Makefile). Remember
   to load the correct modules before building for GPU.

   Try to run the code with different number of GPUs, e.g. 1, 2, 4, 8 and 16 (two nodes).
   Use the same number of MPI tasks and GPUs per node.

   Does the program scale? How does it perform in comparison to the CPU version in the
   ["Scalability"](../scalability) exercise?

2. In the basic version of the program, there is no multi-GPU awareness. Due to that, 
   all the MPI tasks within a node are using the same GPU (id=0) and performance suffers.

   It is possible to overcome this by using a wrapper script, which makes only particular
   GPU visible for each MPI task. Modify the batch job script as follows and try to rerun
   the experiment in the step 1.
   ```
   #SBATCH ...

   cat << EOF > select_gpu
   #!/bin/bash

   export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
   exec \$*
   EOF
   chmod u+x select_gpu

   srun ./select_gpu ./heat_offload
   rm -rf ./select_gpu
   ```

3. Enable multi-GPU awareness in the program by uncommenting in the [Makefile](heat-equation/Makefile) the `SETDEVICE` variable. Now, **do not** use the `select_gpu` wrapper (it does not affect performance, however, number of GPUs reported by the application won't be correct)

4. In LUMI, it is possible to do MPI communications directly from the GPUs. However,
   the basic version of program performs all the MPI communication from CPUs which can
   have significant performance impact. 

   Build a version for GPU aware MPI by uncommenting in the [Makefile](heat-equation/Makefile) the `GPU_MPI` variable. In addition, at runtime one needs to set the environment
   variable `MPICH_GPU_SUPPORT_ENABLED`, so add to the batch job script
   ```
   export MPICH_GPU_SUPPORT_ENABLED=1
   ```

   How is the performance and scalability now? Compare the performance again to the CPU
   version of ["Scalability"](../scalability) exercise.
   
