FC=ftn
GPU_MPI=
# GPU_MPI=-DGPU_MPI
SET_DEVICE=
# SET_DEVICE=-DSETDEVICE

FCFLAGS=-O3 -fopenmp $(GPU_MPI) $(SETDEVICE)
LDFLAGS=-fopenmp 

EXE=heat_openmp
OBJS=main.o heat_mod.o core.o setup.o utilities.o

all: $(EXE)

heat_mod.o: heat_mod.F90
core.o: core.F90 heat_mod.o 
utilities.o: utilities.F90 heat_mod.o
setup.o: setup.F90 heat_mod.o utilities.o
main.o: main.F90 heat_mod.o core.o setup.o utilities.o

$(EXE): $(OBJS)
	$(FC) $(FCFLAGS) $(OBJS) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.mod *.png *.acc.s *~
