import pytest
import numpy as np


from SparseAutoschedulingBenchmark.Benchmarks.PageRank import pagerank
from SparseAutoschedulingBenchmark.BinsparseFormat import BinsparseFormat
from SparseAutoschedulingBenchmark.Frameworks.NumpyFramework import NumpyFramework


@pytest.mark.parametrize(
    "A,expected",
    [
        (np.array([[0, 1],
                   [1, 0]], dtype=float),
         np.array([0.5, 0.5], dtype=float)),

        (np.array([[0, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0]], dtype=float),
         None),

        (np.array([[0, 0],
                   [1, 0]], dtype=float),
         None),
    ],
)
def test_pagerank(A, expected):
    xp = NumpyFramework()

    A_bin = BinsparseFormat.from_numpy(A)

    result_bin = pagerank(xp, A_bin)

    result = xp.from_benchmark(result_bin).ravel()

    if expected is not None:
        assert np.allclose(result, expected, atol=1e-2)
    else:
        assert np.isclose(np.sum(result), 1.0, atol=1e-6)
        assert np.all(result >= 0)

        if A.shape == (3, 3) and np.all(A == np.array([[0, 0, 0],
                                                       [1, 0, 0],
                                                       [0, 1, 0]])):
            eps = 1e-6
            assert (result[2] > result[1] + eps) and (result[1] > result[0] + eps)

        if A.shape == (2, 2) and np.all(A == np.array([[0, 0],
                                                       [1, 0]])):
            eps = 1e-6
            assert result[1] > result[0] + eps