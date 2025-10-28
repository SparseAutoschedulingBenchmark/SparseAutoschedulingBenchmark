import pytest

import numpy as np

from SparseAutoschedulingBenchmark.Benchmarks.Jacobi import benchmark_jacobi
from SparseAutoschedulingBenchmark.BinsparseFormat import BinsparseFormat
from SparseAutoschedulingBenchmark.Frameworks.CheckerFramework import CheckerFramework
from SparseAutoschedulingBenchmark.Frameworks.NumpyFramework import NumpyFramework
from SparseAutoschedulingBenchmark.Frameworks.PyDataSparseFramework import (
    PyDataSparseFramework,
)


@pytest.mark.parametrize(
    "xp, A, b, x",
    [
        (
            PyDataSparseFramework(),
            np.array([[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]]),
            np.array([5.0, 8.0, 8.0]),
            np.zeros((3,)),
        ),
        (
            NumpyFramework(),
            np.array([[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]]),
            np.array([5.0, 8.0, 8.0]),
            np.zeros((3,)),
        ),
        (
            NumpyFramework(),
            np.array(
                [
                    [10.0, 1.0, 0.0, 2.0],
                    [1.0, 8.0, 1.0, 0.0],
                    [0.0, 2.0, 9.0, 1.0],
                    [1.0, 0.0, 1.0, 7.0],
                ]
            ),
            np.array([16.0, 18.0, 15.0, 16.0]),
            np.zeros((4,)),
        ),
        (
            NumpyFramework(),
            np.array([[20.0, 3.0, 1.0], [2.0, 15.0, 4.0], [1.0, 2.0, 18.0]]),
            np.array([24.0, 21.0, 21.0]),
            np.zeros((3,)),
        ),
        (
            CheckerFramework(),
            np.array([[100.0, 1.0, 1.0], [1.0, 100.0, 1.0], [1.0, 1.0, 100.0]]),
            np.array([102.0, 102.0, 102.0]),
            np.zeros((3,)),
        ),
        (
            CheckerFramework(),
            np.array(
                [
                    [12.0, 2.0, 0.0, 0.0, 1.0],
                    [1.0, 10.0, 3.0, 0.0, 0.0],
                    [0.0, 2.0, 11.0, 1.0, 0.0],
                    [0.0, 0.0, 2.0, 13.0, 3.0],
                    [1.0, 0.0, 0.0, 2.0, 14.0],
                ]
            ),
            np.array([17.0, 24.0, 17.0, 31.0, 19.0]),
            np.zeros((5,)),
        ),
    ],
)
def test_jacobi_solver(xp, A, b, x):
    A_bin = BinsparseFormat.from_numpy(A)
    b_bin = BinsparseFormat.from_numpy(b)
    x_bin = BinsparseFormat.from_numpy(x)

    x_test = benchmark_jacobi(xp, A_bin, b_bin, x_bin)
    x_sol = xp.from_benchmark(x_test)
    x_sol = np.round(x_sol, decimals=4)

    b_coo = BinsparseFormat.to_coo(b_bin)
    assert b_coo == BinsparseFormat.to_coo(BinsparseFormat.from_numpy(A @ x_sol))
