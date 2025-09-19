from SparseAutoschedulingBenchmark.Benchmarks.MatMul import benchmark_matmul
from SparseAutoschedulingBenchmark.Frameworks.NumpyFramework import NumpyFramework
from SparseAutoschedulingBenchmark.BinsparseFormat import BinsparseFormat
import numpy as np
import pytest

@pytest.mark.parametrize(
    "A, B",
    [
        (
            np.array([[1, 2], [3, 4]], dtype=np.int64),
            np.array([[1, 0], [0, 1]], dtype=np.int64)
        ),
    ],
)
def test_matmul(A, B):
    bframework = NumpyFramework()

    A_f = BinsparseFormat.from_numpy(A)
    B_f = BinsparseFormat.from_numpy(B)

    C_f = benchmark_matmul(bframework, A_f, B_f)

    C = BinsparseFormat.to_numpy(C_f)

    assert C == A @ B