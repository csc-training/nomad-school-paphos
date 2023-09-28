# Single GPU performance aspects

In this short exercise we illustrate the single most important
performance aspect of GPU usage, *i.e.* data copies between
CPU (host) and GPU (device) memories.

The short program [jacobi.F90](jacobi.F90) solves the
the dimensional Laplace equation numerically with the so called Jacobi
iteration. The key component of the numerical algorithm is stencil update,
which is common operation in numerical methods.

In the algorithm, one performs an iteration where at each iteration the numerical
solution is updated. The provided [Makefile](Makefile) builds three different versions
of the code:

- `jacobi_cpu` : a pure CPU version 
- `jacobi_implicit` : a GU version where data is copied between the host and device 
   at each iteration.
- `jacobi_explicit` : a GPU version where data copied only in the beginning and in the
   end of iterations.

Run the different versions (with single GPU or CPU core) and investigate their
performance characteristics. You may also have a look on the source code to get
a feel of simple OpenMP offloading code.
