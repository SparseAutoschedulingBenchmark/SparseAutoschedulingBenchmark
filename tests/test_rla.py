import pytest

import numpy as np

from sparseappbench.benchmarks.RLA_Benchmark import (
    benchmark_johnson_lindenstrauss_nn,
    data_knn_rla_generator,
)
from sparseappbench.frameworks.numpy_framework import NumpyFramework


def as_ndarray(xp, x):
    """Convert anything from benchmark/lazy format into a proper NumPy array."""
    if hasattr(xp, "from_benchmark"):
        return np.asarray(xp.from_benchmark(x))
    return np.asarray(x)


@pytest.mark.parametrize(
    "n_samples, n_features, k, eps",
    [
        (10, 8, 3, 0.2),
        (20, 15, 5, 0.1),
        (50, 30, 7, 0.3),
    ],
)
def test_benchmark_shapes(n_samples, n_features, k, eps):
    xp = NumpyFramework()

    # Data
    data_bench = xp.random.randn(n_samples, n_features)
    query_bench = xp.random.randn(1, n_features)

    # JL projection
    projection_matrix = data_knn_rla_generator(xp, data_bench, seed=42, eps=eps)

    nearest_ind, nearest_dist = benchmark_johnson_lindenstrauss_nn(
        xp, data_bench, query_bench, projection_matrix, k=k, eps=eps
    )

    # Convert lazy/benchmark arrays to NumPy for assertions
    nearest_ind = xp.compute(nearest_ind)
    nearest_dist = xp.compute(nearest_dist)

    assert nearest_ind.shape == (k,)
    assert nearest_dist.shape == (k,)
    assert (nearest_dist >= 0).all()


def test_jl_preserves_distance_order():
    xp = NumpyFramework()
    n_samples = 20
    n_features = 10
    k = 3
    eps = 0.2

    data_bench = xp.random.randn(n_samples, n_features)
    query_bench = xp.random.randn(1, n_features)

    projection_matrix = data_knn_rla_generator(xp, data_bench, seed=42, eps=eps)

    nearest_ind, _ = benchmark_johnson_lindenstrauss_nn(
        xp, data_bench, query_bench, projection_matrix, k=k, eps=eps
    )

    # Compute concrete arrays for NumPy operations
    nearest_ind = xp.compute(nearest_ind)
    data_np = xp.compute(data_bench)
    query_np = xp.compute(query_bench)

    # True distances
    diff = data_np - query_np
    orig_distances = np.sqrt(np.sum(diff**2, axis=1))
    orig_order = np.argsort(orig_distances)[:k]

    # Check overlap
    overlap = len(set(nearest_ind.astype(int)) & set(orig_order))
    assert overlap >= 1


def test_data_knn_rla_generator_shape_and_scale():
    xp = NumpyFramework()
    n_samples = 10
    n_features = 8
    eps = 0.2

    data_bench = xp.random.randn(n_samples, n_features)
    U = data_knn_rla_generator(xp, data_bench, seed=42, eps=eps)

    # Compute concrete array
    U_array = xp.compute(U)

    target_dim = int(np.log(n_samples) / (eps * eps))
    if target_dim > n_features:
        target_dim = n_features

    assert U_array.shape == (n_features, target_dim)
    assert np.count_nonzero(U_array) > 0
