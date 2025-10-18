import pytest
import numpy as np

from SparseAutoschedulingBenchmark.Benchmarks.CP_ALS import (
    benchmark_cp_als,
    dg_cp_als_sparse_small,
)
from SparseAutoschedulingBenchmark.BinsparseFormat import BinsparseFormat
from SparseAutoschedulingBenchmark.Frameworks.CheckerFramework import CheckerFramework
from SparseAutoschedulingBenchmark.Frameworks.NumpyFramework import NumpyFramework

@pytest.mark.parametrize("xp", [NumpyFramework(), CheckerFramework()])
def test_cp_als_basic(xp):
    """Testing that CP-ALS runs without errors and produces correct output shapes"""
    X_bin, rank, max_iter = dg_cp_als_sparse_small()

    A_bin, B_bin, C_bin, lambda_bin = benchmark_cp_als(xp, X_bin, rank, max_iter = 5)

    # Checking output shapes
    I, J, K = X_bin.data["shape"]
    assert A_bin.data["shape"] == (I, rank)
    assert B_bin.data["shape"] == (J, rank)
    assert C_bin.data["shape"] == (K, rank)
    assert lambda_bin.data["shape"] == (rank,)

    print(f" CP-ALS test passed with {xp.__class__.__name__}")
