# Process/thread affinity

In this exercise you can investigate with a simple test program how processes
and threads are mapped to physical cores. Program performs also simple calculations
and prints out the time spent in the calculation. Calculation is replicated over the
MPI tasks / threads, and there are no parallel overheads so that in principle the timing
should always remain the same.

Example output like
```
...
Rank 002 thread 01 on nid002533 core = 8-11 numa = MPOL_DEFAULT (3.309177 seconds)
...
```
means that operating system is allowed to move the thread in any of the physical cores
8-11. Note: AMD CPUs in LUMI support also 2-way simultaneous multithreading (SMT). When
SMT is enabled, physical cores have ids 0-127 and "hyperthreads" ids 128-255.

1. Build the code with the provided [Makefile](Makefile) by issuing `make`

2. Try to run code in serially and with a basic parallel configuration with the provided
   batch job scripts [job_affinity_serial.sh](job_affinity_serial.sh) and 
   [job_affinity_parallel.sh](job_affinity_parallel.sh). You should see that code runs
   slower when using all the cores in the node due to dynamic scaling of CPU clock 
   frequencies.

3. There are also batch job scripts ("bad" and "very bad") that on purpose mess up the
   mapping between processes/threads to cores. Try to run the code with them and 
   investigate the results.

4. Compare the results for different cases when code is build with different compilers,
   e.g.
   ```
   module swap PrgEnv-cray PrgEnv-gnu
   make -B
   ```

You may play in the batch job scripts with settings for `OMP_NUM_THREADS` and 
try setting there also `export OMP_PLACES=cores`.


