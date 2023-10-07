#!/bin/bash
module load craype-accel-amd-gfx90a rocm

make -B
cp heat_offload heat_offload_1

make -B SET_DEVICE=-DSETDEVICE
cp heat_offload heat_offload_2

make -B SET_DEVICE=-DSETDEVICE GPU_MPI=-DGPU_MPI
cp heat_offload heat_offload_3
