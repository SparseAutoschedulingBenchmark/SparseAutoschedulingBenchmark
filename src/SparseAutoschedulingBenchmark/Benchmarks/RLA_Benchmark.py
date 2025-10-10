

"""
Name: Randomized Numerical Linear Algebra and Algorithms. 
Author: Vilohith Gokarakonda

The purpose of this is to create python tests that are for Randomized Linear Algebra methods.
    Specifically, I will first show the application of the Johnson Lindenstrauss Lemma for Nearest Neighbor. 
    My goal is to write benchmarks on applications of Randomized Numerical Linear Algebra, paticuarily
    for graph algorithms, PDEs, and Scientific Machine Learning.

    Next PR will be from this paper: SPARSE GRAPH BASED SKETCHING FOR FAST NUMERICAL LINEAR ALGEBRA

    My semester goal project will be to understand Learning Green’s functions associated with time-dependent
    partial differential equations and start writing code in Finchlite an understand finch implementation of these methods. 

"""


def benchmark_johnson_lindenstrauss_nn(xp, data_bench, query_bench, k=5, eps=0.1):
    

    # I have seen other PR's do this, but I am still not completely sure of the point of this
    data = xp.lazy(xp.from_benchmark(data_bench))
    query = xp.lazy(xp.from_benchmark(query_bench))

    n_samples, n_features = data.shape

    #  Johnson Lindenstrauss Theorem Lemmna. The eps represents the disortion of distance by epsilon, between the the original space and the reduced subsapce
    target_dim =  xp.log(n_samples) / (eps * eps) 

    if target_dim > n_features:
        target_dim = n_features

    xp.random.seed(40)
    projection_matrix = xp.random.normal(0, 1 / xp.sqrt(target_dim), (n_features, int (target_dim))) # where you create projection matrix based on the lemmna 

    # Project to lower subspace
    projected_data = xp.matmul(data, projection_matrix)
    projected_query = xp.matmul(query, projection_matrix)

    #-----K Nearest Neighbour from here on out--------

    distances = xp.zeros(n_samples)

    # Norm Calculations
    diff = projected_data - projected_query
    distances = xp.sqrt(xp.sum(diff**2, axis = 1))

    nearest_indices = xp.zeros(k)
    nearest_distances = xp.full(k, xp.inf)

    # No need for einsum, since I am only iterating over one dimension of the matrix.
    for i in range(n_samples):
        current_dist = distances[i]
        for j in range(k):
            if current_dist < nearest_distances[j]:
                for l in range(k - 1, j, -1):
                    nearest_distances[l] = nearest_distances[l - 1]
                    nearest_indices[l] = nearest_indices[l - 1]

                nearest_distances[j] = current_dist
                nearest_indices[j] = i
                break

    # Just puts the results in 3 by 5 matrix. nearest_indices is scalar that associates to sample point i in original space. Distance is in projected subspace.  
    result_indices = xp.stack([xp.arange(k), nearest_indices], axis=0)
    result = xp.stack([result_indices, nearest_distances], axis=0)

    result = xp.compute(result)
    return xp.to_benchmark(result)

    
