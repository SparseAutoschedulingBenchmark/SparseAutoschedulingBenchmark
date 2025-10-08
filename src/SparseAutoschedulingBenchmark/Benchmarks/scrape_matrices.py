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
import json


def append_to_json(filename, matrix_name, matrix_group, norm_value, n, nnz):
    """Append matrix name and norm to JSON file."""
    # Try to load existing data, or create empty list if file doesn't exist
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    
    # Append new entry
    data.append({
        "matrix_name": matrix_name,
        "matrix_group": matrix_group,
        "norm": norm_value,
        "n": n,
        "nnz": nnz
    })

    data.sort(key=lambda x: x["norm"])
    
    # Write back to file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def already_in_json(filename, matrix_name):
    """Check if a matrix name is already in the JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            return any(entry["matrix_name"] == matrix_name for entry in data)
    except (FileNotFoundError, json.JSONDecodeError):
        return False

def main():
    parser = argparse.ArgumentParser(description="Scrape matrices from SuiteSparse Matrix Collection")
    parser.add_argument("--limit", type=int, default=100000, help="Maximum number of matrices to retrieve")
    parser.add_argument("--maxsize", type=int, default=100000, help="Maximum matrix nnz to retrieve")
    parser.add_argument("--output", type=str, default="matrix_norms.json", help="Output JSON file for matrix norms")
    args = parser.parse_args()

    # Get all matrices (use a large limit to get the full collection)
    matrices = ssgetpy.search(isspd=True, nzbounds=(0, args.maxsize), limit=args.limit)

    # Take a random permutation
    matrices = random.sample(list(matrices), len(matrices))
    for matrix in matrices:
        (path, archive) = matrix.download(extract=True)
        matrix_path = os.path.join(path, matrix.name + ".mtx")
        print(f"Matrix: {matrix.name}, Path: {matrix_path}")
        if matrix_path and os.path.exists(matrix_path):
            if already_in_json(args.output, matrix.name):
                print(f"Skipping {matrix.name}, already in {args.output}")
                continue
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
                print(f"Norm of (A-D)/D: {norm_value}")
                
                # Write to JSON file
                append_to_json(args.output, matrix.name, matrix.group, float(norm_value), n, A.nnz)
                print(f"Saved {matrix.name} norm to {args.output}")
                
            except Exception as e:
                print(f"Error computing norm for matrix {matrix.name}: {e}")
                continue


if __name__ == "__main__":
    main()
