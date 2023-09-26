---
title:  Parallel programming
event:  Towards exascale solutions in Green function methods and advanced DFT
lang:   en
---

# Parallel programming {.section}

# Programming languages

- The de-facto standard programming languages in HPC are (still!)
  C/C++ and Fortran
- Higher level languages like Python and Julia are gaining popularity
    - Often computationally intensive parts are still written in C/C++
      or Fortran
- Low level GPU programming with CUDA or HIP
- For some applications there are high-level frameworks with
  interfaces to multiple languages
    - SYCL, Kokkos, PETSc, Trilinos
    - TensorFlow, PyTorch for deep learning

# Parallel programming models

![](img/anatomy.svg){.center width=100%}

# Parallel programming models

- Parallel execution is based on threads or processes (or both) which run at the same time on different CPU cores
- Processes
    - Interaction is based on exchanging messages between processes
    - MPI (Message passing interface)
- Threads
    - Interaction is based on shared memory, i.e. each thread can access directly other threads data
    - OpenMP, pthreads

# Parallel programming models

<!-- Copyright CSC -->
 ![](img/processes-threads.svg){.center width=80%}
<div class=column>
**MPI: Processes**

- Independent execution units
- MPI launches N processes at application startup
- Works over multiple nodes
</div>
<div class=column>

**OpenMP: Threads**

- Threads share memory space
- Threads are created and destroyed  (parallel regions)
- Limited to a single node

</div>

# GPU programming models

- GPUs are co-processors to the CPU
- CPU controls the work flow:
  - *offloads* computations to GPU by launching *kernels*
  - allocates and deallocates the memory on GPUs
  - handles the data transfers between CPU and GPUs
- GPU kernels run multiple threads
    - Typically much more threads than "GPU cores"
- When using multiple GPUs, CPU runs typically multiple processes (MPI) or multiple threads (OpenMP)

# GPU programming models

![](img/gpu-offload.svg){.center width=40%}
<br>

- CPU launches kernel on GPU
- Kernel execution is normally asynchronous
    - CPU remains active
- Multiple kernels may run concurrently on same GPU

# Brief introduction to MPI {.section}

# Message-passing interface

- MPI is an application programming interface (API) for distributed parallel
  computing
- MPI programs are portable and scalable
    - the same program can run on different types of computers, from
      laptops to supercomputers
- MPI is flexible and comprehensive
    - large (hundreds of procedures)
    - concise (only 10-20 procedures are typically needed)
- First version of standard (1.0) published in 1994, latest (4.0) in June 2021
    - <https://www.mpi-forum.org/docs/>

# Execution model in MPI

- Normally, parallel program is launched as a set of *independent*, *identical
  processes*
    - execute the *same program code* and instructions
    - processes can reside in different nodes (or even in different computers)
- The way to launch parallel program depends on the computing system
    - **`mpiexec`**, **`mpirun`**, **`srun`**, **`aprun`**, ...
    - **`srun`** on LUMI (and generally when using Slurm batch job system)
- MPI supports also dynamic spawning of processes and launching *different*
  programs communicating with each other
    - rarely used especially with DFT codes

# MPI ranks

<div class="column">
- MPI runtime assigns each process a unique rank (index)
    - identification of the processes
    - ranks range from 0 to N-1
- Processes can perform different tasks and handle different data based
  on their rank
</div>
<div class="column">
```fortran
integer :: a

if (rank == 0) then
   a = 1.0
   ...
else if (rank == 1) then
   a = 0.7
   ...
end if
...
```
</div>

# Data model

- All variables and data structures are local to the process
- Processes can exchange data by sending and receiving messages

![](img/data-model.svg){.center width=100%}

# MPI library

- Information about the communication framework
    - the number of processes
    - the rank of the process
- Communication between processes
    - sending and receiving messages between two or several processes
- Synchronization between processes
- Advanced features
    - Communicator manipulation, user defined datatypes, one-sided communication, ...


# Look and feel of a MPI program

```fortran
program mpi_example
  use mpi_f08
  ...
  call mpi_init(ierr)  ! Initialize MPI
  ! Query number of MPI tasks and rank
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, ierr)
  call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)
  ...
  ! Send messages between two neighboring processes
  left = rank - 1
  right = rank + 1
  call mpi_sendrecv(sendbuf, 100, MPI_DOUBLE, left, tag,  &
    &               recvbuf, 100, MPI_DOUBLE, right, tag, &
    &               MPI_COMM_WORLD, status, ierr)
   ...
  ! Perform element-wise reduction over all MPI tasks
  call mpi_reduce(sendbuf, recvbuf, 100, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
   ...
```

# MPI summary

- In parallel programming with MPI, the key concept is a set of
  independent processes
- Data is always local to the process
- Processes can exchange data by sending and receiving messages
- The MPI library contains functions for communication and
  synchronization between processes


# Brief introduction to OpenMP {.section}

# What is OpenMP?

- A collection of _compiler directives_ and _library routines_ ,
  together with a _runtime system_, for
  **multi-threaded**, **shared-memory parallelization**
- Fortran 77/9X/0X and C/C++ are supported
- Latest version of the standard is 5.2 (November 2021)
    - Full support for accelerators (GPUs)
    - Support latest versions of C, C++ and Fortran
    - Support for a fully descriptive loop construct
    - and more
- Compiler support for 5.0 is still incomplete

# What is OpenMP

- OpenMP parallelized program can be run on your many-core workstation or on a
  node of a cluster
- Enables one to parallelize one part of the program at a time
    - Get some speedup with a limited investment in time
    - Efficient and well scaling code still requires effort
- Serial and OpenMP versions can easily coexist
- GPU programming with OpenMP offloading
- Pure OpenMP program is always limited to a single node
    - Combining with MPI allows to run hybrid MPI/OpenMP programs on multiple nodes


# Three components of OpenMP

- Compiler directives, i.e. language extensions
    - Expresses shared memory parallelization
    - Preceded by sentinel, can compile serial version

- Runtime library routines
    - Small number of library functions
    - Can be discarded in serial version via conditional compiling

- Environment variables
    - Specify the number of threads, thread affinity etc.

# OpenMP directives

- OpenMP directives consist of a *sentinel*, followed by the directive
  name and optional clauses
  <div style="padding-top:1em">

  |         | sentinel      | directive   | clauses          |
  | ------- | ------------- | ----------- | ---------------- |
  | C/C++   | `#pragma omp` | `parallel`  | `private(x,y)`   |
  | Fortran | `!$omp`       | `parallel`  | `private(x,y)`   |

  </div>
- Directives are ignored when code is compiled without OpenMP support


# OpenMP directives

<div class=column>
- In C/C++, a directive applies to the following structured block

```c++
#pragma omp parallel
{
  // calculate in parallel
    printf("Hello world!\n");
}
```
</div>

<div class=column>
- In Fortran, an `end` directive specifies the end of the construct

```fortran
!$omp parallel
  ! calculate in parallel
    write(*,*) "Hello world!"
!$omp end parallel
```
</div>

# Fork-join model

<div class="column">

- Threads are launched (forked) at the start of a *parallel region*
```fortran
  !$omp parallel [clauses]
     structured block
   !$omp end parallel
```
- Prior to it only one thread (main)
- Multiple threads execute the structured block
- After end only main thread continues executiona (join)

</div>
<div class="column">

- Single Program Multiple Data
![](img/omp-parallel.svg){.center width=50%}

</div>


# Look and feel of an OpenMP program

```fortran
program openmp_example
  ...
! Calculate dot product of x and y in parallel
!$omp parallel do shared(x,y,n) private(i) reduction(+:asum)
  do i = 1, n
     asum = asum + x(i)*y(i)
  end do
!$omp end parallel do
```

# OpenMP summary

- OpenMP is an API for thread-based parallelisation in shared memory architectures
- Compiler directives, runtime API, environment variables
- Relatively easy to get started but specially efficient and/or real-world
  parallelisation non-trivial

# GPU programming {.section}

# Introduction to GPUs

- GPUs have become ubiquitous in high-performance computing
- Better FLOPS / W than CPUs
- Single GPU provides high performance
    - Whole LUMI GPU partition = 20 000 GPUs
    - Whole LUMI CPU partition = 200 000 CPU cores
    - GPU partition has ~50 times more performance
- Currently, two players in the market: NVIDIA and AMD

# CPU vs GPU
<div class=column>
**CPU**

- More complex and oriented towards general purpose usage.
- Can run operating systems and very different types of applications
- Better control logic, caches and cache coherence
</div>
<div class=column>
**GPU**

- Large fraction of transistors dedicated to the mathematical operations and less to contorl and caching
- Individual core is less powerful, but there can be
thousands of them
- CPU host is needed for running in the GPU
</div>

# CPU vs GPU

<!-- Image source https://docs.nvidia.com/cuda/cuda-c-programming-guide/ 
     copyright  NVIDIA Corporation -->
![](img/CPU_vs_GPU_alu.png){.center width=80%}

# GPU as co-processor


<div class=column>
- Separate memory spaces for CPU and GPU
- CPU controls the work flow:
  - *offloads* computations to GPU by launching *kernels*
  - allocates and deallocates the memory on GPUs
  - handles the data transfers between CPU and GPUs
</div>

<div class=column>
![](img/cpu-gpu-memory.png){.center width=85%}
</div>

# Programming for GPUs

- Porting application to GPUs is non-trivial
- Algorithms and data structures may need to be adapted (massive amount of parallelism required)
- The physical memory in current GPUs is distinct from CPUs
    - Memory copies between CPU and GPU can easily become bottlenect
- If only part of the application is ported, performance improvement may be modest

# GPU programming approaches

- Directive based approaches: OpenACC and OpenMP
    - "standard" and "portable"
- Native low level languages: CUDA (NVIDIA) and HIP (AMD)
    - HIP supports in principle also NVIDIA devices
    - Fortran needs wrappers via C-bindings
- Performance portability frameworks: SYCL, Kokkos
    - Support only C++
- Standard language features: parallel C++, `do concurrent`
    - Rely on implicit data movements
    - Compiler support incomplete

# Look and feel for OpenMP offloading

```fortran
program offload_example
  ...
! Calculate dot product of x and y in GPU
!$omp target  ! Offload following block to GPU with implicit data movements
!$omp teams distribute parallel do reduction(+:asum)
  do i = 1, n
     asum = asum + x(i)*y(i)
  end do
!$omp end teams distribute parallel do
!$omp end target
```

# Look and feel for CUDA/HIP

```c++
// Calculates y = y + a*x
__global__ void axpy_kernel(int n, float a, float *x, float *y)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) y[tid] += a * x[tid];
}

// x and y need to be "GPU" arrays
void axpy(int n, float a, float *x, float *y)
{
    dim3 blocks(32);
    dim3 threads(256);
    axpy_kernel<<<blocks, threads>>>(n, a, x, y);
}
```

# Look and feel for CUDA/HIP

```c++
// Memory management
hipMalloc((void **) &x_gpu, sizeof(float) * n);
hipMemcpy(x_gpu, x, sizeof(float) * n, hipMemcpyHostToDevice);
...
hipMemcpy(y, y_gpu, sizeof(float) * n, hipMemcpyDeviceToHost);
```
```fortran
interface
  subroutine axpy_gpu(n, a, x, y) bind(C, name='axpy')
    use, intrinsic :: iso_c_binding
    type(c_ptr), value :: x, y
    real(c_float), value :: a
    integer(c_int), value :: n
  end subroutine
end interface
```

# Summary on GPU programming

# Web resources: MPI

- List of MPI functions with detailed descriptions
    - <https://rookiehpc.org/mpi/index.html>
    - <http://mpi.deino.net/mpi_functions/>
- Good online MPI tutorials
    - <https://hpc-tutorials.llnl.gov/mpi/>
    - <http://mpitutorial.com/tutorials/>
    - <https://www.youtube.com/watch?v=BPSgXQ9aUXY>
- MPI coding game in C
    - <https://www.codingame.com/playgrounds/47058/have-fun-with-mpi-in-c/lets-start-to-have-fun-with-mpi>

# Web resources: MPI

- MPI 4.0 standard <http://www.mpi-forum.org/docs/>
- MPI implementations
    - OpenMPI <http://www.open-mpi.org/>
    - MPICH <https://www.mpich.org/>
    - Intel MPI <https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/mpi-library.html>

# Web resources: OpenMP

- OpenMP homepage: <http://openmp.org/>
- Good online reference: <https://rookiehpc.org/openmp/index.html>
- Online tutorials: <http://openmp.org/wp/resources/#Tutorials>

