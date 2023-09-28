FC=ftn
FCFLAGS=-Ofast -fopenmp 
LDFLAGS=-fopenmp

EXE=heat
OBJS=main.o heat_mod.o core.o setup.o utilities.o 

all: $(EXE)

heat_mod.o: heat_mod.F90
core.o: core.F90 heat_mod.o
utilities.o: utilities.F90 heat_mod.o
setup.o: setup.F90 heat_mod.o utilities.o
main.o: main.F90 heat_mod.o core.o setup.o utilities.o

$(EXE): $(OBJS) 
	$(FC) $(FCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.mod *.png *~
