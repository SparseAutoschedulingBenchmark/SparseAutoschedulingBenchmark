"""
Name: diBELLA Transitive Reduction Algorithm
Author: Jaehun Baek and Aaryan Tomar
Email: jbaek90@gatech.edu
Motivation:
This algorithm implements the iterative transitive reduction (TO-DO: Add significance of algorithm)
step from the diBELLA 2D paper
G. Guidi et al., “Parallel String Graph Construction and Transitive Reduction for De Novo Genome Assembly,” 
in Proc. IEEE Int. Parallel & Distributed Processing Symposium (IPDPS), vol. 2021, pp. 517-526, May 2021, 
doi: 10.1109/IPDPS49936.2021.00060.
Role of Sparsity:
The overlap graph R is a sparse matrix where R[i, j]
represents the suffix length of an overlap between read i and read j.
The computation is a sparse matrix-matrix multiplication (SpGEMM)
over the (min, +) semiring, N = R^2, to find shortest 2-hop paths.
Statement on the Use of Generative AI:
No generative AI was used to construct the benchmark function.
This statement was written by hand.  
"""
import numpy as np
from ..BinsparseFormat import BinsparseFormat

def transitive_reduction(xp, R_bench, max_iters = 10):
    """
    Performs iterative transitive reduction on a sparse overlap graph R

    Parameters:
    -----------
    xp : The Array API module
    R_bench : A binsparse tensor representing the overlap matrix R
    max_iters : The maximum number of reduction iterations

    Returns:
    --------
    A binsparse tensor of the transitively reduced graph S
    """

    R_eager = xp.from_benchmark(R_bench)
    R = xp.lazy(R_eager)
    R_nnz_prev = xp.compute(xp.sum(R_eager != 0))[()]
    for i in range(len(max_iters)):
        R_plus = xp.with_fill_value(R, np.inf)
        
        # N <- R ^ 2 in Algo 2 that uses custom MinPlus semiring
        # expressed through einsum- R[i, k] + R[k, j] iterates over all intermediate nodes k, 
        # finds all 2-hop paths, and adds their lengths
        N = xp.einsum("N[i, j] min= R[i, k] + R[k, j]", R=R_plus)

        v = xp.max(R, axis=1)
        # v <- v.APPLY(x.add)

        # Build M matrix
        v_expanded = xp.expand_dims(v, axis=1)
        M_dense = xp.broadcast_to(v_expanded, R.shape)
        R_bool = R != 0
        M = xp.where(R_bool, M_dense, 0)

        N_bool = xp.logical_and(N != 0, N != inf)
        common_sparsity = xp.logical_and(R_bool, N_bool)
        is_transitive = M >= N
        I = xp.logical_and(common_sparsity, is_transitive)

        R = xp.where(I, 0, R)
        R_eager = xp.compute(R)
        R_nnz_new = xp.compute(xp.sum(R_eager != 0))[()]

        if R_nnz_new == R_nnz_prev:
            break

        R_nnz_prev = R_nnz_new
        R = xp.lazy(R_eager)

    return xp.to_benchmark(R_eager)

#TO-DO: add data generator functions