import os

import numpy as np

import SparseAutoschedulingBenchmark as autobench
from SparseAutoschedulingBenchmark.BinsparseFormat import BinsparseFormat
from SparseAutoschedulingBenchmark.Frameworks.AbstractFramework import AbstractFramework
from SparseAutoschedulingBenchmark.Frameworks.einsum import einsum


class NumpyTestFramework(AbstractFramework):
    def __init__(self):
        pass

    def from_benchmark(self, array):
        if array.data["format"] == "dense":
            return np.array(array.data["values"]).reshape(array.data["shape"])
        if array.data["format"] == "COO":
            indices = []
            idx_dim = 0
            while "indices_" + str(idx_dim) in array.data:
                indices.append(array.data["indices_" + str(idx_dim)])
                idx_dim += 1
            V = array.data["values"]
            shape = array.data["shape"]
            data = np.zeros(shape, dtype=V.dtype)
            data[tuple(indices)] = V
            return data
        raise ValueError("Unsupported format: " + array.data["format"])

    def to_benchmark(self, array):
        return BinsparseFormat.from_numpy(array)

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def einsum(self, prgm, **kwargs):
        return einsum(np, prgm, **kwargs)

    def with_fill_value(self, array, value):
        return array

    def __getattr__(self, name):
        return getattr(np, name)


frameworks = {"NumpyTestFramework": NumpyTestFramework()}


def test_main(tmp_path):
    autobench.main(
        frameworks=frameworks,
        framework_names=["NumpyTestFramework"],
        results_folder=tmp_path,
        args=[],  # Empty list to avoid using sys.argv
    )
    assert os.path.exists(tmp_path / "NumpyTestFramework_matmul_matmul_dense_large.csv")
    assert os.path.exists(tmp_path / "NumpyTestFramework_matmul_matmul_dense_small.csv")
    assert os.path.exists(
        tmp_path / "NumpyTestFramework_matmul_matmul_sparse_large.csv"
    )
    assert os.path.exists(
        tmp_path / "NumpyTestFramework_matmul_matmul_sparse_small.csv"
    )
