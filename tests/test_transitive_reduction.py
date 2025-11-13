import pytest
import numpy as np

from SparseAutoschedulingBenchmark.Benchmarks.TransitiveReduction import transitive_reduction
from SparseAutoschedulingBenchmark.BinsparseFormat import BinsparseFormat
from SparseAutoschedulingBenchmark.Frameworks.NumpyFramework import NumpyFramework

def create_graph(xp, edges, n):
    """Helper to build a Binsparse graph from a list of (row, col, val) tuples."""
    rows = [e[0] for e in edges]
    cols = [e[1] for e in edges]
    vals = [e[2] for e in edges]
    
    dense = np.zeros((n, n))
    dense[rows, cols] = vals
    
    return xp.to_benchmark(dense)

def to_dense(xp, bench_matrix):
    """Helper to convert output back to dense array for easy assertion."""
    return xp.from_benchmark(bench_matrix)

def test_case_1_simple_removal():
    """Test that direct edge is removed when indirect path (2-hop) is shorter (better) than the direct edge."""
    xp = NumpyFramework()
    n = 3
    edges = [
        (0, 1, 10.0),
        (1, 2, 10.0),
        (0, 2, 30.0)
    ]
    R_input = create_graph(xp, edges, n)
    
    # Run reduction
    R_output = transitive_reduction(xp, R_input, x=1, max_iters=5)
    output_dense = to_dense(xp, R_output)

    # Assertions
    assert output_dense[0, 1] == 10.0, "Edge 0->1 should be kept"
    assert output_dense[1, 2] == 10.0, "Edge 1->2 should be kept"
    assert output_dense[0, 2] == 0.0,  "Edge 0->2 should be removed (Transitive)"


def test_case_2_keep_better_direct():
    """Test that direct edge is kept when direct edge is shorter (better) than the 2-hop path (or indirect is longer)"""
    xp = NumpyFramework()
    n = 3
    edges = [
        (0, 1, 10.0),
        (1, 2, 10.0),
        (0, 2, 15.0)
    ]
    R_input = create_graph(xp, edges, n)
    
    R_output = transitive_reduction(xp, R_input, x=1, max_iters=5)
    output_dense = to_dense(xp, R_output)

    assert output_dense[0, 2] == 15.0, "Edge 0->2 should be kept (since its the shortest path)"


def test_case_3_keep_bad_path():
    """Test that direct edge is kept whenthe 2-hop path is very long (poor overlap)"""
    xp = NumpyFramework()
    n = 3
    edges = [
        (0, 1, 40.0),
        (1, 2, 40.0),
        (0, 2, 30.0)
    ]
    R_input = create_graph(xp, edges, n)
    
    R_output = transitive_reduction(xp, R_input, x=1, max_iters=5)
    output_dense = to_dense(xp, R_output)

    assert output_dense[0, 2] == 30.0, "Edge 0->2 should be kept (since 2-hop path is worse)"

def test_case_4_equality_removal():
    """Test that direct edge is removed when it is exactly equal in weight (length) to an indirect path."""
    xp = NumpyFramework()
    n = 3
    edges = [
        (0, 1, 10.0),
        (1, 2, 10.0),
        (0, 2, 20.0)
    ]
    R_input = create_graph(xp, edges, n)
    
    R_output = transitive_reduction(xp, R_input, x=1, max_iters=5)
    output_dense = to_dense(xp, R_output)

    assert output_dense[0, 2] == 0.0, "Edge 0->2 should be removed (since equal weight, we should remove redundancy)"