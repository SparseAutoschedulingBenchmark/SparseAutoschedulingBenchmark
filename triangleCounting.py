import numpy as np

def triangle_count_adj(A): # A is adgacency matrix 
   
    n = A.shape[0] # n is num of nodes / num of rows in matrix
    triangle_count = 0

    # go through each node
    for u in range(n):
        # find neighbors of u
        neighbors_u = [v for v in range(n) if A[u, v] == 1] #this means u and v are connected

        for v in neighbors_u:
            if u < v:  # only count each edge once
                # find neighbors of v
                neighbors_v = [w for w in range(n) if A[v, w] == 1] #do same thing as u but for v

                # count common neighbors
                for w in neighbors_u:
                    if w in neighbors_v:
                        triangle_count += 1 #if theres w in both u and v then theres a triangle

    # each triangle counted 3 times (once per edge)
    triangle_count //= 3
    return triangle_count

#Example?
def test_triangle():
    A = np.array([[0,1,1],
                  [1,0,1],
                  [1,1,0]], dtype=int)
    return A

A = test_triangle()
print(triangle_count_adj(A))
