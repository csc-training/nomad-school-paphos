FC=ftn
OMPFLAGS=-fopenmp

EXE=hello_gpu

$(EXE): hello.F90
	$(FC) -o $@ $< $(OMPFLAGS)

.PHONY: clean
clean:
	rm -f $(EXE)
