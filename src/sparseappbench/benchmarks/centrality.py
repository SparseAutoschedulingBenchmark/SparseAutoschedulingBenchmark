"""
Name: Betweenness Centrality

Author: Aadharsh Rajkumar

Email: arajkumar34@gatech.edu

What does this code do: The current code is only able to do the first step of
the process which involves going layer by layer from each potential starting node.
This will allow us to find the number of shortest paths to each other node. The
second step which is not implemented yet will handle the backtracing and finding
the number of times each node is in a shortest path.

Citation for reference implementation: 
https://github.com/SparseApplicationBenchmark/SparseApplicationBenchmark/pull/48/files

Citation for importance of the problem: 
Matta, J., Ercal, G. & Sinha, K. Comparing the speed and accuracy of
approaches to betweenness centrality approximation.
Comput Soc Netw 6, 2 (2019). https://doi.org/10.1186/s40649-019-0062-5

Statement on the use of Generative AI: No generative AI was used to construct
the benchmark function. This statement is written by hand.
"""

def betweenness_centrality(xp, A_binsparse):
    G = xp.lazy(xp.from_benchmark(A_binsparse))
    n = G.shape[0]
    bc_scores = xp.zeros((n,), dtype = float)

    # Will calculate in two steps
    # The first step will take each node and find the amount of shortest paths to each other node.
    # So for example 4 -> 6 could have 3 diff shortest paths and 4 -> 2 could have only 1 shortest path.
    # After we find all these shortest paths we will backtrace them to evaluate each node that lies in the path. 
    # number of times this node is in one of the shortest path divided by total shortest paths between the 
    # two edge nodes gets added to the intermediate nodes bc score.

    for v in range(n):
        number_of_paths = xp.zeros((n,), dtype = float)
        self_dist = xp.zeros((n,), dtype = float)
        self_dist = self_dist + xp.array([1.0 if i == v else 0.0 for i in range(n)])
        number_of_paths = number_of_paths + self_dist

        neighbors = xp.array(G[v], dtype = float)
        layer_traversal = []
        depth = 0

        node_count = xp.compute(xp.sum(neighbors))

        while(node_count != 0):
            depth += 1
            layer_traversal.append(neighbors)
            number_of_paths = number_of_paths + neighbors

            not_neighbors = xp.equal(number_of_paths, 0)
            neighbors = xp.matmul(neighbors, G) * not_neighbors

            node_count = xp.compute(xp.sum(neighbors))


    return