FC=ftn
OMPFLAGS=-fopenmp
LIBS=
LDFLAGS=-lcraymp

EXE1=jacobi_cpu
EXE2=jacobi_implicit
EXE3=jacobi_explicit

all: $(EXE1) $(EXE2) $(EXE3)

$(EXE1): jacobi.F90
	$(FC) -o $@ $< $(LIBS) $(LDFLAGS)

$(EXE2): jacobi.F90
	$(FC) -o $@ $< $(LIBS) $(OMPFLAGS)

$(EXE3): jacobi.F90
	$(FC) -o $@ $< $(LIBS) $(OMPFLAGS) -DMAP_DATA

.PHONY: clean
clean:
	rm -f $(EXE1) $(EXE2) $(EXE3) *.o *.s
