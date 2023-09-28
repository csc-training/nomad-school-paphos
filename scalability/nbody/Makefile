CC=CC
LIBS=
LDFLAGS=-fopenmp
CFLAGS=-DNDEBUG -Ofast -DUSE_MYVECTOR

EXE1=nbody

all: $(EXE1)

$(EXE1): nbody.cpp myvector.hpp
	$(CC) $(CFLAGS) -o $@ $< $(LIBS) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(EXE1)
