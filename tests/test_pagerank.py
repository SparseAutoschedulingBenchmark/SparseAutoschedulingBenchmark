import pytest

import numpy as np

import networkx as nx

from SparseAutoschedulingBenchmark.Benchmarks.PageRank import pagerank
from SparseAutoschedulingBenchmark.BinsparseFormat import BinsparseFormat
from SparseAutoschedulingBenchmark.Frameworks.NumpyFramework import NumpyFramework


@pytest.mark.parametrize(
    "A,expected",
    [
        (np.array([[0, 1], [1, 0]], dtype=float), np.array([0.5, 0.5], dtype=float)),
        (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float), None),
        (np.array([[0, 0], [1, 0]], dtype=float), None),
    ],
)
def test_basic_pagerank_cases(A, expected):
    xp = NumpyFramework()

    A_bin = BinsparseFormat.from_numpy(A)

    result_bin = pagerank(xp, A_bin)

    result = xp.from_benchmark(result_bin).ravel()

    if expected is not None:
        assert np.allclose(result, expected, atol=1e-2)
    else:
        assert np.isclose(np.sum(result), 1.0, atol=1e-6)
        assert np.all(result >= 0)

        if A.shape == (3, 3) and np.all(
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) == A
        ):
            eps = 1e-6
            assert (result[0] < result[1] - eps) and (result[1] < result[2] - eps)

        if A.shape == (2, 2) and np.all(np.array([[0, 0], [1, 0]]) == A):
            eps = 1e-6
            assert result[0] < result[1] - eps


def test_pagerank_against_networkx():
    xp = NumpyFramework()
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)])
    A = nx.to_numpy_array(G, dtype=float)
    A_bin = BinsparseFormat.from_numpy(A)

    result_bin = pagerank(xp, A_bin)
    result = xp.from_benchmark(result_bin).ravel()

    expected_dict = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
    expected = np.array([expected_dict[i] for i in range(len(G))])

    assert np.allclose(result, expected, atol=1e-2)
