# Scalability testing

In this exercise you can investigate with simple programs scalability aspects
of different type of problems. Heat equation solver here is extremely memory bound,
whereas the N-body simulation is largely compute bound.

## Memory bound problem: heat equation

1. Build the code under [heat-equation](heat-equation) with the provided 
  [Makefile](heat-equation/Makefile).

2. Run the code with different number of MPI tasks and OpenMP threads within a
   single node and investigate the performance.

   AMD CPUs in LUMI are divided into so called ["Core Complex Die"s (CCDs)](https://docs.lumi-supercomputer.eu/hardware/lumic/#cpu), 
   where each CCD has 8 cores. CPU cores within a CCD share L3 cache and memory 
   controller, which means that when fewer cores per CCD are used, single core 
   may have more memory bandwidth and L3 cache available.

   Slurm places the MPI process `--cpus-per-task` cores apart, so when running with
   less MPI tasks than there are cores, the option can be used for spreading the MPI 
   tasks apart.

   Compare the performance e.g. with the following Slurm settings:

   ```
   #SBATCH --ntasks-per-node=16
   #SBATCH --cpus-per-task=1

   export OMP_NUM_THREADS=1
   ```

   vs.

   ```
   #SBATCH --ntasks-per-node=16
   #SBATCH --cpus-per-task=8

   export OMP_NUM_THREADS=1
   ```

   Placement of OpenMP threads (when having less threads than cores) can be affected
   by setting environment variable `OMP_PROC_BIND`. OpenMP runtime can also print out
   information about mapping of processes/threads to cores by setting the environment 
   variable `OMP_DISPLAY_AFFINITY`.

   At the end, you should observe that with this memory bound problem best performance
   with a single node is obtained when using only 1-2 cores per CCD, *i.e.* 16-32 cores
   per node in total.

3. Pick some `--ntasks-per-node` and `----cpus-per-task` setting, and investigate the
   scalability with multiple nodes, use *e.g.* one, two, or four nodes. How does the
   scalability look in comparison to the scalability within a node? Can you explain the
   results?

At the end you should have learned that for memory bound problems the way 
MPI processes / OpenMP threads of the program are distributed within a node can have 
huge impact on the perfomance. Also, number of MPI tasks or number of cores is 
necessarily not meaningful variable when discussing scalability. In modern CPUs like
AMD EPYCs performance e.g. with 128 cores can vary drastically depending on whether.
they are distributed to one, two or four nodes.

## Compute bound problem: N-body simulation

1. Build the code under [nbody](nbody) with the provided 
  [Makefile](nbody/Makefile).

2. Run the code with different number of MPI tasks and OpenMP threads within a
   single node, as well as with few nodes and investigate the performance and 
   scalability. With single core the calculation takes over 10 minutes, so you might
   want to start e.g. with four cores.

   You should observe that here the placement of process/threads either to same CCD or
   different CCDs has only a minor effect, *i.e.* available memory bandwidth per core
   is not major performance aspect.

   You should observe also that OpenMP threading is more beneficial here than with the 
   heat equation.
   
3. Try to utilize simultaneous multithreading (SMT) by setting in your Slurm batch job 
   script

   ```
   #SBATCH --hint=multithread
   ```

   Note that there are now 256 logical cores, so you should have `ntasks-per-node * cpus-per-task = 256`. Does SMT benefit this application? (You may try SMT also with the 
   heat equation).


## Bonus task: your favourite DFT code

Try to do some simple performance and scalability experiments with some DFT code. 
Remember that scalability depends always a lot the input case and type of calculation
(number of atoms / basis functions, LDA/GGA vs. hybrid functionals vs. GW), so one should
always carry out at least some scalability testing before starting production runs.

In terms of memory bound / compute bound behaviour DFT codes are typically somewhere 
between the heat equation and N-body problem. Parts of the DFT problem are more memory 
bound (e.g. FFTs) and parts more compute bound (dense linear algebra). 
The main bottleneck depends on the system size and type of calculation, e.g. few tens of
atoms with GGA might spent most time in FFTs, whereas large GW calculation may be more
compute bound.
