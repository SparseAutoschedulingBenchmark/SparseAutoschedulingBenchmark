"""
Name: Markov Clustering Algorithm
Author: Prateek Hanumappanahalli
Email: phanumap3@gatech.edu
Motivation: "The Markov Clustering (MCL) algorithm relies heavily on repeated matrix operations, particularly
matrix multiplication during the expansion step. Since the efficient execution of matrix-based kernels has been 
extensively studied in  linear algebra, MCL serves as an effective benchmark for evaluating the performance of 
iterative numerical methods."
Gene H. Golub and Charles F. Van Loan, Matrix Computations (4th ed., SIAM, 2013), doi: 10.1137/1.9780898719918.
Role of sparsity: The input is a sparse adjacency matrix. 
The algorithm uses sparse matrix multiplication and element-wise operations repeatedly, 
so it depends heavily on efficient sparse matrix functions. 
Implementation: Handwritten code based on the implementation from https://github.com/GuyAllard/markov_clustering.
Data Generation: Data generator functions are not yet developed. This comment will be updated once they are implemented.
Statement on Generative AI: No generative AI was used to write the benchmark function itself. Generative
AI was used to debug code. This statement was written by hand.
"""


import numpy as np
import scipy.sparse as sp

from ..BinsparseFormat import BinsparseFormat

def _normalize(array_api, matrix):
    col_sums = array_api.sum(matrix, axis=0)
    col_sums = array_api.maximum(col_sums, array_api.finfo(matrix.dtype).eps)
    return matrix / col_sums

def _sparse_allclose(array_api, matrix_a, matrix_b, rtol = 1e-5, atol = 1e-8):
    return array_api.all(array_api.abs(matrix_a - matrix_b) <= atol + rtol * array_api.abs(matrix_b))

def _prune(array_api, matrix, threshold):
    pruned_matrix = matrix.copy()
    pruned_matrix[pruned_matrix < threshold] = 0
    
    col_indices = array_api.arange(matrix.shape[1])
    row_indices = array_api.argmax(matrix, axis=0)

    if hasattr(row_indices, 'get'):
         row_indices = row_indices.get()
    
    pruned_matrix[row_indices, col_indices] = matrix[row_indices, col_indices]
    
    if hasattr(pruned_matrix, 'eliminate_zeros'):
        pruned_matrix.eliminate_zeros()
    
    return pruned_matrix

"""

benchmark_mcl(array_api, graph_binsparse, expansion=2, inflation=2, loop_value=1, iterations=100, 
                    pruning_threshold=1e-5, pruning_frequency=1, convergence_check_frequency=1)

Computes Markov Clustering on a given sparse adjacency matrix

Args:
----
array_api: The array API module to utilize
graph_binsparse: The sparse adjacency matrix of the graph in binsparse format.
expansion: The cluster expansion factor.
inflation: The cluster inflation factor.
loop_value: The value to add to the diagonal for self loops.
iterations: The maximum number of iterations.
pruning_threshold: Threshold below which matrix elements will be set to 0.
pruning_frequency: Perform pruning every 'pruning_frequency' iterations.
convergence_check_frequency: Perform convergence check every 'convergence_check_frequency' iterations.

Returns
-------
The final converged matrix in binsparse format

"""
def benchmark_mcl(array_api, graph_binsparse, expansion=2, inflation=2, loop_value=1, iterations=100, 
                    pruning_threshold=1e-5, pruning_frequency=1, convergence_check_frequency=1):
    
    graph_lazy = array_api.lazy(array_api.from_benchmark(graph_binsparse))
    
    loops_matrix = array_api.identity(graph_lazy.shape[0], dtype=graph_lazy.dtype)
    current_matrix = graph_lazy + loop_value * loops_matrix

    current_matrix = _normalize(array_api, current_matrix)
    
    for i in range(iterations):
        previous_matrix = current_matrix
        
        expanded_matrix = current_matrix
        for _ in range(expansion - 1):
            expanded_matrix = array_api.matmul(expanded_matrix, current_matrix)
        
        inflated_matrix = array_api.power(expanded_matrix, inflation)
        current_matrix = _normalize(array_api, inflated_matrix)
        
        if pruning_threshold > 0 and i % pruning_frequency == (pruning_frequency - 1):
            current_matrix = _prune(array_api, current_matrix, pruning_threshold)
        
        if i % convergence_check_frequency == (convergence_check_frequency - 1):
            if _sparse_allclose(array_api, current_matrix, previous_matrix):
                break

    final_matrix = array_api.compute(current_matrix)
    return array_api.to_benchmark(final_matrix)

    

    
    
