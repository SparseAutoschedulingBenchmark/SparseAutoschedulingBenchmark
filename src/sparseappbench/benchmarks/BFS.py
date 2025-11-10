"""
Name: Breadth-First Search (BFS)

Co-Authors: Aarav Joglekar, Joel Mathew Cherian

Reference:
J. Kepner and J. Gilbert (eds.), “Graph Algorithms in the Language of Linear Algebra,”
Society for Industrial and Applied Mathematics (SIAM), Philadelphia, 2011.

Motivation:
This benchmark expresses the BFS algorithm as linear algebra operations using einsums.
It was translated from the Julia Finch Einsum implementation.

Statement on the use of Generative AI:
No generative AI was used to write the benchmark function itself. Generative AI was used
to assist in translation and code explanation. This statement was written by hand.
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
    while frontier_count > 0:
        level = xp.where(frontier, level_idx, level)
        visited = xp.logical_or(visited, frontier)
        frontier = xp.einsum(
            "frontier[j] += edges[i,j] * frontier[i]", edges=edges, frontier=frontier
        )
        frontier = xp.logical_and(frontier, xp.logical_not(visited))
        frontier_count = xp.sum(frontier)
        level_idx += 1
    return xp.to_benchmark(level)
