"""
Name: CP-ALS for Tensor Decomposition
Author: Grace Wang
Email: gwang426@gatech.edu
Motivation:
"The Alternating Least Squares (ALS) algorithm for CANDECOMP/PARAFAC (CP) plays a critical
role in tensor decomposition, which has applications in various fields such as signal processing,
pscyhometrics, neuroscience, and graph analysis. Within the ALS algorithm, the Matricized-Tensor
Times Khatri-Rao Product (MTTKRP) operation is a computationally intensive step that often
dominates the overall runtime. Efficiently implementing MTTKRP is crucial for the performance of 
the ALS algorithm."
T. G Kolda and B. W. Bader, "Tensor Decompositions and Applications," SIAM Review, vol. 51, no. 3, 
p. 455-500, 2009, doi: 10.1137/07070111X.
Role of sparsity:
The input tensor is sparse, and the ALS algorithm takes advantage of this sparsity through its 
MTTKRP kernel, which process only the non-zero elements of the tensor. For sparse tensors with 
nnz << I * J * K, the complexity reduces from O(I * J * K * R) to O(nnz * R), where nnz is the 
number of non-zero elements in the tensor and R is the decomposition rank. This makes it practical
to work with large-scale applications.
Implementation:
Handwritten code based on the standard CP-ALS algorithm from:
T. G Kolda and B. W. Bader, "Tensor Decompositions and Applications," SIAM Review, vol. 51, no. 3, 
p. 455-500, 2009, doi: 10.1137/07070111X.
Reference implementations used and translated:
https://github.com/willow-ahrens/18.335FinalProject/blob/submit/TFGDCANDECOMP.py
https://github.com/tensorly/tensorly/blob/main/tensorly/decomposition/_cp.py
Data Generation: 
Sparse tensors with controlled rank and sparsity were generated using random tensor creation methods
from Tensortools combined with sparsification techniques from TensorLy.
Statement on the use of Generative AI:
No generative AI was used to construct the benchmark function itself. Generative AI was used to debug
some parts of the code. This statement was written by hand.
"""
import numpy as np
import scipy.sparse as sp

from ..BinsparseFormat import BinsparseFormat

"""
benchmark_cp_als(xp, X_bench, rank, max_iter)

Computes the CP decomposition using Alternating Least Squares (ALS).
Factorizes a 3rd-order tensor X into factor matrices A, B, C such that:
$X \approx \sum_{r=1}^{R} \lambda_r \cdot a_r \circ b_r \circ c_r$

where $\circ$ denotes the outeer product, R is the rank, and $\lambda$ are the weights.

Args:
----
xp: The array API module to use
X_bench: The input 3rd-order sparse tensor in binsparse format
rank: Number of components (rank) for the decomposition
max_iter: Maximum number of ALS iterations

Returns:
-------
Tuple of (A_bench, B_bench, C_bench, lambda_bench) in binsparse format where:
- A, B, and C are the normalized factor matrices
- lambda are the component weights
"""
def benchmark_cp_als(xp, X_bench, rank, max_iter=50):
    X_eager = xp.from_benchmark(X_bench)
    X = xp.lazy(X_eager)

    I, J, K = X_bench.data["shape"]
    dtype = X_bench.data["values"].dtype

    np.random.seed(0)
    A = xp.lazy(xp.from_benchmark(BinsparseFormat.from_numpy(np.random.rand(I, rank).astype(dtype))))
    B = xp.lazy(xp.from_benchmark(BinsparseFormat.from_numpy(np.random.rand(J, rank).astype(dtype))))
    C = xp.lazy(xp.from_benchmark(BinsparseFormat.from_numpy(np.random.rand(K, rank).astype(dtype))))

    for iteration in range(max_iter):

        # Update A
        mttkrp_result = xp.einsum("mttkrp_result[i, r] += X[i, j, k] * B[j, r] * C[k, r]", X = X, B = B, C = C)
        CtC = xp.einsum("CtC[r, s] += C[k, r] * C[k, s]", C = C)
        BtB = xp.einsum("BtB[r, s] += B[j, r] * B[j, s]", B = B)
        G = xp.multiply(xp.compute(CtC), xp.compute(BtB))
        G_pinv = np.linalg.pinv(xp.compute(G))
        A_eager = xp.matmul(xp.compute(mttkrp_result), G_pinv)
        A = xp.lazy(A_eager) # Converting back to lazy for next iteration

        # Update B
        mttkrp_result = xp.einsum("mttkrp_result[j, r] += X[i, j, k] * A[i, r] * C[k, r]", X = X, A = A, C = C)
        AtA = xp.einsum("AtA[r, s] += A[i, r] * A[i, s]", A = A)
        G = xp.multiply(xp.compute(CtC), xp.compute(AtA))
        G_pinv = np.linalg.pinv(xp.compute(G))
        B_eager = xp.matmul(xp.compute(mttkrp_result), G_pinv)
        B = xp.lazy(B_eager) 

        # Update C
        mttkrp_result = xp.einsum("mttkrp_result[k, r] += X[i, j, k] * A[i, r] * B[j, r]", X = X, A = A, B = B)
        BtB = xp.einsum("BtB[r, s] += B[j, r] * B[j, s]", B = B)
        G = xp.multiply(xp.compute(BtB), xp.compute(AtA))
        G_pinv = np.linalg.pinv(xp.compute(G))
        C_eager = xp.matmul(xp.compute(mttkrp_result), G_pinv)
        C = xp.lazy(C_eager)

    # Normalizing factors
    A_norms_sq = xp.einsum("norms[r] += A[i, r] * A[i, r]", A = A)
    B_norms_sq = xp.einsum("norms[r] += B[j, r] * B[j, r]", B = B)
    C_norms_sq = xp.einsum("norms[r] += C[k, r] *C[k, r]", C = C)

    A_norms = xp.compute(xp.sqrt(xp.compute(A_norms_sq)))
    B_norms = xp.compute(xp.sqrt(xp.compute(B_norms_sq)))
    C_norms = xp.compute(xp.sqrt(xp.compute(C_norms_sq)))

    # Compute lambda
    lambda_vals = xp.multiply(xp.multiply(A_norms, B_norms), C_norms)

    A_norms_2d = xp.expand_dims(A_norms, 0)
    B_norms_2d = xp.expand_dims(B_norms, 0)
    C_norms_2d = xp.expand_dims(C_norms, 0)

    # Case: avoiding division by zero
    eps = 1e-10
    A_norms_safe = xp.maximum(A_norms_2d, eps)
    B_norms_safe = xp.maximum(B_norms_2d, eps)
    C_norms_safe = xp.maximum(C_norms_2d, eps)
    
    A_normalized = xp.divide(xp.compute(A), A_norms_safe)
    B_normalized = xp.divide(xp.compute(B), B_norms_safe)
    C_normalized = xp.divide(xp.compute(C), C_norms_safe)

    # Convert to binsparse format
    A_bench_out = xp.to_benchmark(A_normalized)
    B_bench_out = xp.to_benchmark(B_normalized)
    C_bench_out = xp.to_benchmark(C_normalized)
    lambda_bench_out = xp.to_benchmark(lambda_vals)

    return (A_bench_out, B_bench_out, C_bench_out, lambda_bench_out)

# Data generators
def dg_cp_als_sparse_small():
    np.random.seed(0)

    # Create sparse 3rd-order tensor
    I, J, K = 20, 20, 20
    rank = 3
    nnz = int(0.01 * I * J * K)

    # Generate random sparse tensor indices (no duplicates)
    all_indices = np.random.choice(I * J * K, size = nnz, replace = False)
    i_idx, j_idx, k_idx = np.unravel_index(all_indices, (I, J, K))

    values = np.random.rand(nnz).astype(np.float32)
    X_bin = BinsparseFormat.from_coo((i_idx, j_idx, k_idx), values, (I, J, K))
    max_iter = 50

    return (X_bin, rank, max_iter)

def dg_cp_als_factorizable_small():
    """
    Generating a small factorizable tensor by creating random factor matrices
    and reconstructing the tensor from them (tensor should decompose easily 
    with low reconstruction error).
    """
    np.random.seed(0)
    I, J, K = 20, 20, 20
    rank = 3

    A = np.random.rand(I, rank).astype(np.float32)
    B = np.random.ran(J, rank).astype(np.float32)
    C = np.random.rand(K, rank).astype(np.float32)

    X_dense = np.einsum('ir,jr,kr->ijk', A, B, C)
    indices = np.nonzero(X_dense)
    values = X_dense[indices]
    X_bin = BinsparseFormat.from_coo(indices, values, (I ,J, K))
    max_iter = 50

    return (X_bin, rank, max_iter)