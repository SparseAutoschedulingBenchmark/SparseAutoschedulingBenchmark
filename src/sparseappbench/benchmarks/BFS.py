"""
Name: Breadth-First Search (BFS)
Co-Authors: Aarav Joglekar, Joel Mathew Cherian
Email: ajoglekar32@gatech.edu

Motivation: The Bredth-First Search algorithm is an important graph traversal technique
used to epxlore vertices by layers. It is a fundamental building block for more complex
graph algorithms, especially in areas like parallel processing and high-performance
computing. It is often parallelized to run on GPUs for speed.

Role of sparsity:
In standard BFS, algorithms on sparse graphs are faster because they process fewer
edges, and specialized algebraic methods use sparsity to avoid unnecessary computations
by focusing only on non-zero elements. Optimizing the use of sparse data structures and
algorithms is key to achieving high performance, as it reduces memory footprint and
leads to faster traversals.

Implementation Reference:
J. Kepner and J. Gilbert (eds.), “Graph Algorithms in the Language of Linear Algebra,”
Society for Industrial and Applied Mathematics (SIAM), Philadelphia, 2011.

Data Generation:
Graphs for this benchmark may be created manually for testing or generated
procedurally using a Graph500-style R-MAT model. Generated adjacency matrices
are converted into the benchmark format with BinsparseFormat.from_numpy().
"""


def benchmark_bfs(xp, adjacency_matrix, source):
    """
    Returns level id of vertices in a graph during BFS from
    a given source vertex.
    """
    edges = xp.from_benchmark(adjacency_matrix)
    (n, m) = edges.shape
    assert n == m
    visited = xp.zeros((n,), dtype=bool)
    frontier = xp.zeros((n,), dtype=bool)
    frontier[source] = True
    level = xp.zeros((n,), dtype=int)
    level_idx = 1
    frontier_count = 1

    edges = xp.lazy(edges)
    visited = xp.lazy(visited)
    frontier = xp.lazy(frontier)
    level = xp.lazy(level)
    while frontier_count > 0:
        level = xp.where(frontier, level_idx, level)
        visited = xp.logical_or(visited, frontier)
        frontier = xp.einsum(
            "frontier[j] += edges[i,j] * frontier[i]", edges=edges, frontier=frontier
        )
        frontier = xp.logical_and(frontier, xp.logical_not(visited))
        visited = xp.compute(visited)
        frontier = xp.compute(frontier)
        level = xp.compute(level)
        frontier_count = xp.sum(frontier)
        frontier_count = xp.compute(frontier_count)
        level_idx += 1
    return xp.to_benchmark(level)
