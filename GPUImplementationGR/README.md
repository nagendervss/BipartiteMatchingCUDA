Implementation of the algorithm described in paper *A Push-Relabel-Based Maximum Cardinality Bipartite Matching Algorithms on GPUs*

This module implements *bipartite matching* with *group relabeling* as described in the paper.

Please run the Makefile to compile the code using the following command

```
$ make
```

The input file to this implementation should be in the Matrix Market File Format. The code can be run using the provided sample data file with the following command

```
$ ./bipartiteMatching ../SampleData/divorceMM/divorce.mtx
```
