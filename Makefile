# nanollama.c - build variants
#
#   make            -> optimized build (O3; add -march=native locally if you like)
#   make portable   -> portable build, no CPU-specific flags (for CI / other machines)
#   make avx2       -> explicit AVX2+FMA code paths
#   make debug      -> unoptimized with symbols, for gdb
#
# No external dependencies, no OpenMP: the thread pool is hand-rolled in
# src/threads.c (Windows threads on _WIN32, pthreads everywhere else).
#
ifeq ($(OS),Windows_NT)
    EXE := nanollama.exe
else
    EXE := nanollama
endif

CC      = gcc
SRC     := src/main.c src/transformer.c src/tokenizer.c src/sampler.c src/threads.c
CORE    := -std=c11 -Wall -Wextra
LIBS    := -lm

.PHONY: all portable avx2 debug clean

all: $(EXE)

$(EXE): $(SRC) src/nanollama.h
	$(CC) $(CORE) -O3 $(SRC) -o $(EXE) $(LIBS)

portable: $(SRC) src/nanollama.h
	$(CC) $(CORE) -O2 $(SRC) -o $(EXE) $(LIBS)

avx2: $(SRC) src/nanollama.h
	$(CC) $(CORE) -O3 -mavx2 -mfma $(SRC) -o $(EXE) $(LIBS)

debug: $(SRC) src/nanollama.h
	$(CC) $(CORE) -O0 -g $(SRC) -o $(EXE) $(LIBS)

clean:
	rm -f $(EXE)
