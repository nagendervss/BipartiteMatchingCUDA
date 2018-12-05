INCPATHS=/usr/local/cuda/include/
LIBPATHS=/usr/local/cuda/lib64
INCPATHSKERNELS=./

all: bipartiteMatching.o bipartiteMatchingKernels.o
	nvcc bipartiteMatching.o bipartiteMatchingKernels.o -o bipartiteMatching

bipartiteMatching.o: bipartiteMatching.cpp
	gcc -c -o bipartiteMatching.o bipartiteMatching.cpp

bipartiteMatchingKernels.o: bipartiteMatchingKernels.cu bipartiteMatchingKernels.h
	nvcc -I $(INCPATHSKERNELS) -c -o bipartiteMatchingKernels.o bipartiteMatchingKernels.cu
