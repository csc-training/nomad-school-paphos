CC=cc
LIBS=-lm -lnuma
LDFLAGS=-fopenmp
CFLAGS=

EXE=check_affinity

$(EXE): cpu_affinity.c utilities.c
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(EXE)
