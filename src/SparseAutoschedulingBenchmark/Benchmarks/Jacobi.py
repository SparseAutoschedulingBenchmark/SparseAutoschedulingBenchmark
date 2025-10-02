import numpy as np
from scipy.io import mmread
from scipy.sparse import random

from ..BinsparseFormat import BinsparseFormat

"""
Name: Jacobi Iterative Solver
Author: Benjamin Berol
Email: bberol3@gatech.edu
Motivation:
"There exist a wide variety of iterative methods, with the two main classes being the
stationary iterative methods (Young, 1971), such as Richardson's method and Jacobi's
method, and Krylov subspace methods (Liesen and Strakos, 2012) such as the conjugate
gradient method (CG; Hestenes and Stiefel, 1952)."
J. Cockayne, I. C. F. Ipsen, C. J. Oates, and T. W. Reid,
“Probabilistic iterative methods for linear systems,”
J. Mach. Learn. Res., vol. 22, p. 1-34, 2021.
[Online]. Available: https://www.jmlr.org/papers/volume22/21-0031/21-0031.pdf
Role of Sparsity:
Sparsity makes the Jacobi method efficient because each update only needs to
access the nonzero entries, reducing complexity from O(mn) to O(nnz)
Implementation:
Hand-written code modelling the algorithm structure outlined in:
https://www.cs.princeton.edu/~appel/papers/jacobi.pdf
Data Generation:
Data collected from SuiteSparse Matrix Collection that is diagonally
dominant and invertible
Statement on the use of Generative AI:
No generative AI was used to write the benchmark function itself. Generative
AI was used to debug code. This statement was written by hand.
"""


def benchmark_jacobi(
    xp, A_bench, b_bench, x_bench, rel_tol=1e-8, abs_tol=1e-20, max_iters=100
):
    A = xp.lazy(xp.from_benchmark(A_bench))
    b = xp.lazy(xp.from_benchmark(b_bench))
    x = xp.lazy(xp.from_benchmark(x_bench))

    tolerance = max(xp.compute(xp.lazy(rel_tol) * norm(xp, b))[()], abs_tol)
    d = xp.diagonal(A)
    if xp.compute(xp.any(d == 0)):
        raise ValueError("Jacobi requires nonzero diagonal entries.")

    r = b - A @ x
    it = 0

    while xp.compute(norm(xp, r))[()] >= tolerance and it < max_iters:
        x = x + r / d
        x = xp.lazy(xp.compute(x))

        r = b - xp.matmul(A, x)
        r = xp.lazy(xp.compute(r))
        it += 1
    if it >= max_iters:
        raise RuntimeError(
            "Jacobi did not converge within the maximum number of iterations"
        )
    x_solution = xp.compute(x)
    return xp.to_benchmark(x_solution)


def norm(xp, v):
    norm = xp.sqrt(xp.sum(xp.multiply(v, v)))
    if norm == xp.inf:
        raise RuntimeError("Matrix is diverging to infinity")
    return norm


def dg_jacobi_sparse_small():
    rng = np.random.default_rng(0)
    A = mmread("./InputData/Jacobi/nos1.mtx")
    check_convergence(A)
    x = random(
        A.shape[1], 1, density=0.1, format="coo", dtype=np.float64, random_state=rng
    )
    b = (A @ x).tocoo()
    x = np.zeros(
        A.shape[1],
    )

    A_bin = BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)
    b_bin = BinsparseFormat.from_coo((b.row, b.col), b.data, b.shape)
    x_bin = BinsparseFormat.from_numpy(x)
    return (A_bin, b_bin, x_bin)


def check_convergence(A):
    A = A.tocsr()
    diag = np.abs(A.diagonal())
    off_diag_sum = np.abs(A).sum(axis=1).A1 - diag
    is_diag_dom = diag > off_diag_sum
    if np.all(is_diag_dom):
        print("Matrix is diagonally dominant: True")
    else:
        print("Matrix is diagonally dominant: False")
