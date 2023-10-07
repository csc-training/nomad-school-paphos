# GPU profiling with rocprof

In this exercise we examine performance behaviour of multi-GPU program with `rocprof`.
As a a test program you can use the [heat-equation](../multiple-gpu-performance/heat-equation) in the [Multiple GPU](..../multiple-gpu-performance) hands-on.

`rocprof` is standard command line profiler included in **rocm** installation, which can 
produce both a flat profile, and a trace of GPU kernels and HIP API calls. `rocprof` 
does not profile any CPU code or MPI calls, just GPU functions.

In order to use `rocprof` no recompilation is needed, it is enough to start the
application with `rocprof`. As `rocprof` has no knowledge about MPI, for multi-GPU
programs one typically needs launch `rocprof` via wrapper script, so that output from 
different MPI processes is written to different files.

1. You can work with existing build of [heat-equation](../multiple-gpu-performance/heat-equation). For profiling, you can update the batch job script as follows:
   ``` 
   #SBATCH ...

   cat << EOF > rocprof_wrapper.sh
   #!/bin/bash

   exec rocprof --hip-trace -o results_\${SLURM_PROCID}.csv \$*
   EOF

   chmod +x ./rocprof_wrapper.sh
   srun ./rocprof_wrapper.sh ./heat_offload
   rm -f ./rocprof_wrapper.sh
   ```

   After run you should have a set of "results_x" files. "results_0.stats.csv" contains 
   the flat profile of process 0, and "results_0.json" the trace for process 0.
   Traces can be investigated with web browser:
     - Download the trace file(s) (e.g. "results_0.json") to your laptop
     - Go with a browser (chrome recommended, firefox should also work) to https://ui.perfetto.dev/
     - Open the trace file.
   You see the list of perfetto controls by typing `?`

   Try to compare traces for cases both with and without GPU aware MPI. (You can provide different base filename for the two cases and open them e.g. to two tabs/windows
   
