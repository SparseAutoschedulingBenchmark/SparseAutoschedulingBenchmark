import numpy as np
import scipy as sp

"""
Name: Random Numerical Linear Algenra
Author: Vilohith Gokarakonda
Email: vgokarakonda3@gatech.edu
Motivation (Importance of problem with citation):
The purpose of this is to create python tests that are for RLA methods.
Specifically, I will first show the application of the JL Lemma for NN.
My goal is to write benchmarks on applications of RNLA,
for graph algorithms, PDEs, and Scientific Machine Learning

Murray, R., Demmel, J., Mahoney, M. W., Erichson, N. B.,
Melnichenko, M., Malik, O. A., ... & Dongarra, J. (2023).
Randomized numerical linear algebra: A perspective on the field with an eye to software.
arXiv preprint arXiv:2302.11474.
Role of sparsity (How sparsity is used in the problem):
The inputs to the matrix multiply are sparse.
Implementation (Where did the reference algorithm come from? With citation.):
Hand-written, direct call to array api function
https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html
Data Generation (How is the data generated? Why is it realistic?):
Sparse-sparse matrix multiplication is sensitive to sparsity patterns and their
interaction. We use random sparsity patterns for now.  Statement on the use of
Generative AI: No generative AI was used to construct the benchmark function
itself. Generative AI might have been used to construct tests. This statement
was written by hand.
"""


def benchmark_johnson_lindenstrauss_nn(
    xp, data_bench, query_bench, projection_matrix, k=5, eps=0.1
):
    data = xp.lazy(xp.from_benchmark(data_bench))
    query = xp.lazy(xp.from_benchmark(query_bench))

    n_samples, n_features = data.shape
    #  Johnson Lindenstrauss Theorem Lemmna.
    # The eps represents the disortion of distance by epsilon,
    # between the the original space and the reduced subspace
    target_dim = np.log(n_samples) / (eps * eps)
    if target_dim > n_features:
        target_dim = n_features

    # final_projection_matrix = xp.array(projection_matrix_sparse_scaled.toarray())

    # Project to lower subspace
    projected_data = xp.matmul(data, projection_matrix)
    projected_query = xp.matmul(query, projection_matrix)

    # -----K Nearest Neighbour from here on out--------

    # Euclidean distances
    diff = projected_data - projected_query
    distances = xp.sqrt(xp.sum(diff**2, axis=1))

    # Get nearest k neighbors
    nearest_indices = xp.argsort(distances)[:k]
    nearest_distances = xp.sort(distances)[:k]

    # Just puts the results in 3 by k matrix. nearest_indices is scalar
    # that associates to sample point i in original space.
    # Distance is in projected subspace.

    result_indices = xp.stack([xp.arange(k), nearest_indices], axis=0)
    result = xp.stack([result_indices, nearest_distances], axis=0)

    result = xp.compute(result)
    return xp.to_benchmark(result)


def data_knn_rla_generator(xp, data_bench, seed=40, eps=0.1):
    data = xp.lazy(xp.from_benchmark(data_bench))

    n_samples, n_features = data.shape
    target_dim = np.log(n_samples) / (eps * eps)
    if target_dim > n_features:
        target_dim = n_features

    s = np.sqrt(n_features)  # s = 1/density
    density = 1.0 / s  # probability of a nonzero entry = density.
    density_half = density / 2.0  # probability for + or -
    scale = np.sqrt(s / target_dim)  # scale = sqrt(s / n_components)

    rng = np.random.default_rng(seed)

    U_Neg = sp.sparse.random(
        n_features,
        target_dim,
        density_half,
        data_rvs=lambda k: np.full(k, -scale),
        random_state=rng,
    )
    U_Pos = sp.sparse.random(
        n_features,
        target_dim,
        density_half,
        data_rvs=lambda k: np.full(k, scale),
        random_state=rng,
    )
    U = U_Neg + U_Pos
    return xp.to_benchmark(U.toarray())
