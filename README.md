# BipartiteMatchingCUDA
Bipartite Matching Implementations

Implementation of the paper *A Push-Relabel-Based Maximum Cardinality Bipartite Matching Algorithm on GPUs*.

The implementation is split into different modules which implement the basic version of bipartite matching, bipartite matching with global relabeling and bipartite matching with work lists. The modules are as follows

> GPUImplementation : Basic version of bipartite matching as described in the paper<br/>
> GPUImplementationGR : Bipartite Matching with global relabeling as described in the paper<br/>
> GPUImplementationWL : Bipartite Matching with work lists as described in the paper<br/>

Please look into the *README* files of each module for more details.

The test graphs to be used with these implementations should be in a file with Matrix Market File Format.

The source of sample data present in the repository is [divorce laws](https://sparse.tamu.edu/Pajek/divorce) in [SuiteSparse Matrix Collection](https://sparse.tamu.edu/)
