import pytest

import numpy as np

from SparseAutoschedulingBenchmark.Frameworks.NumpyFramework import NumpyFramework


@pytest.fixture
def xp():
    return NumpyFramework()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_basic_addition_with_transpose(xp, rng):
    """Test basic addition with transpose"""
    A = rng.random((5, 5))
    B = rng.random((5, 5))

    C = xp.einsum("C[i,j] = A[i,j] + B[j,i]", A=A, B=B)
    C_ref = A + B.T

    assert np.allclose(C, C_ref)


def test_matrix_multiplication(xp, rng):
    """Test matrix multiplication using += (increment/accumulation)"""
    A = rng.random((3, 4))
    B = rng.random((4, 5))

    C = xp.einsum("C[i,j] += A[i,k] * B[k,j]", A=A, B=B)
    C_ref = A @ B

    assert np.allclose(C, C_ref)


def test_element_wise_multiplication(xp, rng):
    """Test element-wise multiplication"""
    A = rng.random((4, 4))
    B = rng.random((4, 4))

    C = xp.einsum("C[i,j] = A[i,j] * B[i,j]", A=A, B=B)
    C_ref = A * B

    assert np.allclose(C, C_ref)


def test_sum_reduction(xp, rng):
    """Test sum reduction using +="""
    A = rng.random((3, 4))

    C = xp.einsum("C[i] += A[i,j]", A=A)
    C_ref = np.sum(A, axis=1)

    assert np.allclose(C, C_ref)


def test_maximum_reduction(xp, rng):
    """Test maximum reduction using max="""
    A = rng.random((3, 4))

    C = xp.einsum("C[i] max= A[i,j]", A=A)
    C_ref = np.max(A, axis=1)

    assert np.allclose(C, C_ref)


def test_outer_product(xp, rng):
    """Test outer product"""
    A = rng.random(3)
    B = rng.random(4)

    C = xp.einsum("C[i,j] = A[i] * B[j]", A=A, B=B)
    C_ref = np.outer(A, B)

    assert np.allclose(C, C_ref)


def test_batch_matrix_multiplication(xp, rng):
    """Test batch matrix multiplication using +="""
    A = rng.random((2, 3, 4))
    B = rng.random((2, 4, 5))

    C = xp.einsum("C[b,i,j] += A[b,i,k] * B[b,k,j]", A=A, B=B)
    C_ref = np.matmul(A, B)

    assert np.allclose(C, C_ref)


def test_minimum_reduction(xp, rng):
    """Test minimum reduction using min="""
    A = rng.random((3, 4))

    C = xp.einsum("C[i] min= A[i,j]", A=A)
    C_ref = np.min(A, axis=1)

    assert np.allclose(C, C_ref)
