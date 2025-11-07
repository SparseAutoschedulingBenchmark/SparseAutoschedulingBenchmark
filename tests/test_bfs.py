import numpy as np

from SparseAutoschedulingBenchmark.Benchmarks.BFS import benchmark_bfs
from SparseAutoschedulingBenchmark.BinsparseFormat import BinsparseFormat
from SparseAutoschedulingBenchmark.Frameworks.NumpyFramework import NumpyFramework


def test_bfs_basic():
    xp = NumpyFramework()

    A = np.array(
        [
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
        ],
        dtype=bool,
    )

    source = 0
    expected = np.array([1, 2, 2, 3, 3, 4], dtype=int)

    A_bin = BinsparseFormat.from_numpy(A)
    result_bin = benchmark_bfs(xp, A_bin, source)
    result = xp.from_benchmark(result_bin).ravel()

    assert np.array_equal(result, expected), (
        f"BFS output mismatch.\nGot {result}, expected {expected}"
    )
