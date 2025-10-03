import numpy as np
from ..BinsparseFormat import BinsparseFormat

def benchmark_dibella2d(xp, A_bench):
    A_lazy = xp.lazy(xp.from_benchmark(A_bench))
    A_lazy_transpose = A_lazy.T
    C_lazy = xp.matmul(A_lazy, A_lazy_transpose)
    C_lazy_aligned = alignment(C_lazy)
    R_lazy = prune(C_lazy_aligned)
    S_lazy = transitive_reduction(R_lazy)

    S_eager = xp.compute(S_lazy)
    return xp.to_benchmark(S_eager)


# unsure how to go about implementing these
def alignment(m):
    return m

def prune(m):
    return m

def transitive_reduction(m):
    return m