"""
Name: Bellman Ford Algorithm
Author: Ilisha Gupta, Joel Mathew Cherian
Email: igupta90@gatech.edu

What does this code do:
This code implements an Array-API compatible version of Bellman Ford Algorithm
to find the shortest distance from a src node to all edges across a graph.
It takes in an adjacency matrix as an input and then slowly relaxes each vector
by broadcasting it and then determining the minimum distances iteratively.

Role of sparsity:
Often, many graphs when represented as adjacency matrices do not have a lot of
edges while there are n^2 entries, which means that sparsity can be large and optimised.

Citation for reference implementation:
Kepner, Jeremy, and John Gilbert, eds.
Graph algorithms in the language of linear algebra

Statement on the use of Generative AI: No generative AI was used to construct
the benchmark function itself. This statement was written by hand.
"""


def bellman_ford(xp, edges, src):
    # Clarification – should I add lazy/compute here too?
    # edges = xp.lazy(edges)
    edges = xp.from_benchmark(edges)
    n = edges.shape[0]

    G = xp.asarray(edges, dtype=float)
    D = xp.full((n,), xp.inf)
    D[src] = 0

    G, D = xp.compute((G, D))

    for _ in range(n):
        candidates = D[:, None] + G
        D = xp.minimum(D, candidates.min(axis=0))
    return xp.to_benchmark(D)
