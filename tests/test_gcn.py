import pytest

import numpy as np

from SparseAutoschedulingBenchmark.BinsparseFormat import BinsparseFormat
from SparseAutoschedulingBenchmark.Frameworks.CheckerFramework import CheckerFramework
from SparseAutoschedulingBenchmark.Frameworks.NumpyFramework import NumpyFramework

from SparseAutoschedulingBenchmark.Benchmarks.GCN import benchmark_gcn


def gcn_reference(adjacency, features, weights1, bias1, weights2, bias2):
    """Reference implementation of 2-layer GCN"""
    # Layer 1: adjacency @ features -> linear transform -> ReLU
    h1 = adjacency @ features
    h1 = h1 @ weights1 + bias1
    h1 = np.maximum(h1, 0)  # ReLU activation
    
    # Layer 2: adjacency @ h1 -> linear transform  
    h2 = adjacency @ h1
    output = h2 @ weights2 + bias2
    
    return output


@pytest.mark.parametrize(
    "xp,adjacency,features,weights1,bias1,weights2,bias2",
    [
        (
            NumpyFramework(),
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),  # Simple graph adjacency
            np.array([[1, 0], [0, 1], [1, 1]]),  # 3 nodes, 2 features each
            np.array([[1, 0], [0, 1]]),  # 2x2 weight matrix
            np.array([0, 0]),  # bias vector
            np.array([[1], [1]]),  # 2x1 weight matrix for output
            np.array([0]),  # output bias
        ),
        (
            NumpyFramework(),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # Identity adjacency
            np.array([[1, 2], [3, 4], [5, 6]]),  # Different feature values
            np.array([[0.5, 0.5], [0.5, 0.5]]),  # Averaging weights
            np.array([1, 1]),  # Non-zero bias
            np.array([[2], [3]]),  # Different output weights
            np.array([1]),  # Non-zero output bias
        ),
        (
            NumpyFramework(),
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),  # Zero adjacency
            np.array([[1, 1], [2, 2], [3, 3]]),  # Uniform features
            np.array([[1, 0], [0, 1]]),  # Identity weights
            np.array([0, 0]),  # Zero bias
            np.array([[1], [1]]),  # Sum weights
            np.array([0]),  # Zero output bias
        ),
        (
            CheckerFramework(),
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),  # Simple graph adjacency
            np.array([[1, 0], [0, 1], [1, 1]]),  # 3 nodes, 2 features each
            np.array([[1, 0], [0, 1]]),  # 2x2 weight matrix
            np.array([0, 0]),  # bias vector
            np.array([[1], [1]]),  # 2x1 weight matrix for output
            np.array([0]),  # output bias
        ),
        (
            CheckerFramework(),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # Identity adjacency
            np.array([[1, 2], [3, 4], [5, 6]]),  # Different feature values
            np.array([[0.5, 0.5], [0.5, 0.5]]),  # Averaging weights
            np.array([1, 1]),  # Non-zero bias
            np.array([[2], [3]]),  # Different output weights
            np.array([1]),  # Non-zero output bias
        ),
        (
            CheckerFramework(),
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),  # Zero adjacency
            np.array([[1, 1], [2, 2], [3, 3]]),  # Uniform features
            np.array([[1, 0], [0, 1]]),  # Identity weights
            np.array([0, 0]),  # Zero bias
            np.array([[1], [1]]),  # Sum weights
            np.array([0]),  # Zero output bias
        ),
    ],
)
def test_benchmark_gcn(xp, adjacency, features, weights1, bias1, weights2, bias2):
    output_ref = gcn_reference(adjacency, features, weights1, bias1, weights2, bias2)

    adjacency_bin = BinsparseFormat.from_numpy(adjacency)
    features_bin = BinsparseFormat.from_numpy(features)
    weights1_bin = BinsparseFormat.from_numpy(weights1)
    bias1_bin = BinsparseFormat.from_numpy(bias1)
    weights2_bin = BinsparseFormat.from_numpy(weights2)
    bias2_bin = BinsparseFormat.from_numpy(bias2)

    output_bin = benchmark_gcn(xp, adjacency_bin, features_bin, weights1_bin, bias1_bin, weights2_bin, bias2_bin)
    output_bin = BinsparseFormat.to_coo(output_bin)

    assert output_bin == BinsparseFormat.to_coo(BinsparseFormat.from_numpy(output_ref))