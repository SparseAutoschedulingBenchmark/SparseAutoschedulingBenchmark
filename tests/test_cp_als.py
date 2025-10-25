import pytest
import numpy as np

from SparseAutoschedulingBenchmark.Benchmarks.CP_ALS import (
    benchmark_cp_als,
    dg_cp_als_sparse_small,
    dg_cp_als_factorizable_small,
)
from SparseAutoschedulingBenchmark.BinsparseFormat import BinsparseFormat
from SparseAutoschedulingBenchmark.Frameworks.CheckerFramework import CheckerFramework
from SparseAutoschedulingBenchmark.Frameworks.NumpyFramework import NumpyFramework

@pytest.mark.parametrize("xp", [NumpyFramework(), CheckerFramework()])
def test_cp_als_basic(xp):
    """Testing that CP-ALS runs without errors and produces correct output shapes"""
    X_bin, rank, max_iter = dg_cp_als_sparse_small()

    A_bin, B_bin, C_bin, lambda_bin = benchmark_cp_als(xp, X_bin, rank, max_iter = max_iter)

    # Checking output shapes
    I, J, K = X_bin.data["shape"]
    assert A_bin.data["shape"] == (I, rank)
    assert B_bin.data["shape"] == (J, rank)
    assert C_bin.data["shape"] == (K, rank)
    assert lambda_bin.data["shape"] == (rank,)

    print(f" CP-ALS test passed with {xp.__class__.__name__}")

@pytest.mark.parametrize("xp", [NumpyFramework()])
def test_cp_als_reconstruction_error(xp):
    """Tests that CP-ALS produces low reconstruction error on a factorizable tensor"""
    X_bin, rank, max_iter = dg_cp_als_factorizable_small()

    A_bin, B_bin, C_bin = benchmark_cp_als(xp, X_bin, rank, max_iter = max_iter)
    X = xp.from_benchmark(X_bin)
    A = xp.from_benchmark(A_bin)
    B = xp.from_benchmark(B_bin)
    C = xp.from_benchmark(C_bin)

    Y = xp.einsum("Y[i, j, k] += A[i, l] * B[j, l] * C[k, l]", A=A, B=B, C=C)
    Y_computed = xp.compute(Y)

    X_norm_sq = xp.einsum("norm[s] += X[i, j, k] * X[i, j, k]", X=X)
    X_norm = xp.compute(xp.sqrt(xp.compute(X_norm_sq)))[0]

    diff = xp.substract(Y_computed, X)
    diff_norm_sq = xp.einsum("norm[s] += diff[i, j, k] * diff[i, j, k]", diff=diff)
    diff_norm = xp.compute(xp.sqrt(xp.compute(diff_norm_sq)))[0]

    rel_error = diff_norm / X_norm
    print(f"Reconstruction relative error: {rel_error:.6f}")
    assert rel_error < 0.1, f"Reconstruction error too high: {rel_error:.6f}"
    print(f"CP-ALS reconstruction error test passed (error={rel_error:.6f})")

@pytest.mark.parametrize("xp", [NumpyFramework(), CheckerFramework()])
def test_cp_als_factorizable_basic(xp):
    """Test CP-ALS on factorizable tensor (basic shape check)"""
    X_bin, rank, max_iter = dg_cp_als_factorizable_small()

    A_bin, B_bin, C_bin, lambda_bin = benchmark_cp_als(xp, X_bin, rank, max_iter=max_iter)
    I, J, K = X_bin.data["shape"]
    assert A_bin.data["shape"] == (I, rank)
    assert B_bin.data["shape"] == (J, rank)
    assert C_bin.data["shape"] == (K, rank)
    assert lambda_bin.data["shape"] == (rank,)
    print(f"CP-ALS factorizable test passed with {xp.__class__.__name__}")