import numpy as np
import scipy.sparse as sp

from SparseAutoschedulingBenchmark.Benchmarks.mcl_benchmark import benchmark_mcl 
from SparseAutoschedulingBenchmark.BinsparseFormat import BinsparseFormat
from SparseAutoschedulingBenchmark.Frameworks.NumpyFramework import NumpyFramework


def get_cluster_count(matrix):
    if not sp.isspmatrix(matrix):
        matrix = sp.csc_matrix(matrix)
    elif not sp.isspmatrix_csc(matrix):
        matrix = matrix.tocsc()

    attractors = matrix.diagonal().nonzero()[0]
    
    clusters = set()
    for attractor in attractors:
        cluster_indices = matrix.getrow(attractor).nonzero()[1].tolist()
        clusters.add(tuple(sorted(cluster_indices)))

    return len(clusters)


def test_mcl_example():
    
    xp = NumpyFramework()

    A = np.array([
        [0, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.float32)

    expected_count = 2

    A_sparse = sp.coo_matrix(A)
    A_bin = BinsparseFormat.from_coo((A_sparse.row, A_sparse.col), A_sparse.data, A_sparse.shape)
    
    result_bin = benchmark_mcl(xp, A_bin, expansion=2, inflation=2, loop_value=1)
    
    result_matrix = xp.from_benchmark(result_bin)
    
    actual_count = get_cluster_count(result_matrix)

    assert actual_count == expected_count, ( f"MCL failed. Found {actual_count} clusters, but expected {expected_count}." )