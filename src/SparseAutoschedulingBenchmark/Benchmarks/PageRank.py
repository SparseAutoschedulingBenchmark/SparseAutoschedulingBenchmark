"""
Name: Google Page Rank Algorithm

Author: Aarav Joglekar

Email: ajoglekar32@gatech.edu

What does this code do:
First the code calls from_binsparse on the wrapper to translate from binsparse COO. 
Once that is done the out-degree of the adjacency is found by summing columns, giving 
us the number of outbound links per page. Then we duplicate the matrix A (since we
don't want to override its values) and normalize its columns to form the transition 
matrix M. If out-degree is not 0, we divide by k (the number of outbound links). 
If out-degree is 0, that means the node had no links, so we distribute it evenly among 
all nodes to preserve probability mass. We then run  iteration multiple times so 
that the PageRank vector converges to its theoretical stationary value.

Citation for reference implementation: 
“Page Rank Algorithm and Implementation.” GeeksforGeeks, GeeksforGeeks, 15 Apr. 2025, www.geeksforgeeks.org/python/page-rank-algorithm-implementation/. 

Citation corroborating the choice of data & importance of problem:
Brin, S., & Page, L. (1998). "The anatomy of a large-scale hypertextual Web search engine." *Computer Networks and ISDN Systems*, 30(1-7), 107-117.

Statement on the use of Generative AI:
AI was not used in order to write this benchmark function. 
AI might have been used for the purposes of genreating 
test cases for the algorithm that was implemented, picking
up any unfamiliar python syntax, and cleaning up written documentation
flow. 
"""
import numpy as np

from SparseAutoschedulingBenchmark.BinsparseFormat import BinsparseFormat

def pagerank(xp, A_binsparse, alpha = 0.85, max_iter = 100, tol = 1e-6):
    A = xp.from_benchmark(A_binsparse)
    out_degree = xp.sum(A, axis = 0)
    M = xp.array(A, dtype = float)
    N = A.shape[0]
    for j in range(N):
        if out_degree[j] != 0:
            M[:, j] = M[:, j] / out_degree[j]
        else:
            M[:, j] = 1.0 / N
    x = xp.full((N,), 1.0/N)
    u = xp.full((N,), 1.0/N)
    for _ in range(max_iter):
        x_new = alpha * xp.matmul(M, x) + (1 - alpha) * u
        if xp.linalg.norm(x_new - x, ord=1) < tol:
            break
        x = x_new
    return xp.to_benchmark(x)

