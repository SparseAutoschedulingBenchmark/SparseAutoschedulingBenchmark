import pytest

import numpy as np

from sparseappbench.benchmarks.bellmanford import bellman_ford
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.numpy_framework import NumpyFramework


@pytest.mark.parametrize(
    "xp,edges,src,expected",
    [
        (
            NumpyFramework(),
            np.array([[np.inf, 1.0], [np.inf, np.inf]], dtype=float),
            0,
            np.array([0.0, 1.0], dtype=float),
        ),
        (
            NumpyFramework(),
            np.array(
                [[np.inf, 1.0, 4.0], [np.inf, np.inf, 2.0], [np.inf, np.inf, np.inf]],
                dtype=float,
            ),
            0,
            np.array([0.0, 1.0, 3.0], dtype=float),
        ),
        (
            NumpyFramework(),
            np.array(
                [
                    [np.inf, 1.0, np.inf],
                    [np.inf, np.inf, np.inf],
                    [np.inf, np.inf, np.inf],
                ],
                dtype=float,
            ),
            0,
            np.array([0.0, 1.0, np.inf], dtype=float),
        ),
        (
            NumpyFramework(),
            np.array(
                [
                    [np.inf, 1.0, np.inf],
                    [np.inf, np.inf, 2.0],
                    [np.inf, np.inf, np.inf],
                ],
                dtype=float,
            ),
            1,
            np.array([np.inf, 0.0, 2.0], dtype=float),
        ),
        (
            NumpyFramework(),
            np.array(
                [
                    [np.inf, 5.0, 2.0, 10.0],
                    [np.inf, np.inf, 1.0, 3.0],
                    [np.inf, np.inf, np.inf, 2.0],
                    [np.inf, np.inf, np.inf, np.inf],
                ],
                dtype=float,
            ),
            0,
            np.array([0.0, 5.0, 2.0, 4.0], dtype=float),
        ),
    ],
)
def test_bellman_ford(xp, edges, src, expected):
    edges_bin = BinsparseFormat.from_numpy(edges)
    result_bin = bellman_ford(xp, edges_bin, src)
    result = xp.from_benchmark(result_bin).ravel()
    assert np.allclose(result, expected, atol=1e-6, equal_nan=True)
