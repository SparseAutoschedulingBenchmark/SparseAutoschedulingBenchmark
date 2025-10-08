#!/usr/bin/env python3
"""
Simple script to iterate over all matrices in the SuiteSparse Matrix Collection using ssgetpy.
"""

import ssgetpy
import argparse
from scipy.io import mmread
import scipy as sp
import os
import random


def main():
    parser = argparse.ArgumentParser(description="Scrape matrices from SuiteSparse Matrix Collection")
    parser.add_argument("--limit", type=int, default=10000, help="Maximum number of matrices to retrieve")
    args = parser.parse_args()

    # Get all matrices (use a large limit to get the full collection)
    matrices = ssgetpy.search(isspd=True, nzbounds=(0, 10000), limit=args.limit)

    # Take a random sample
    sample_size = min(10000, len(matrices))
    sampled_matrices = random.sample(list(matrices), sample_size)
    
    for matrix in sampled_matrices:
        (path, archive) = matrix.download(extract=True)
        matrix_path = os.path.join(path, matrix.name + ".mtx")
        print(f"Matrix: {matrix.name}, Path: {matrix_path}")
        if matrix_path and os.path.exists(matrix_path):
            A = mmread(matrix_path)  # This is the full sparse matrix
            (m, n) = A.shape
            if m != n:
                print(f"Skipping non-square matrix {matrix.name} of shape {A.shape}")
                continue
            # Convert to CSR format if needed for better diagonal access
            if not hasattr(A, 'diagonal'):
                A = A.tocsr()
            
            
            # Extract diagonal and create diagonal matrix
            d = A.diagonal()
            D = sp.sparse.diags(d, format='csr')
            
            # Calculate the norm
            try:
                norm_value = sp.sparse.linalg.norm((A - D)/d, 2)
            except Exception as e:
                print(f"Error computing norm for matrix {matrix.name}: {e}")
                continue
            print(f"Norm of (A-D)/D: {norm_value}")


if __name__ == "__main__":
    main()
