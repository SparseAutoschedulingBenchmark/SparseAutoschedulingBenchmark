import sparse as sp

from ..BinsparseFormat import BinsparseFormat
from .AbstractFramework import AbstractFramework
from .einsum import einsum


class PyDataSparseFramework(AbstractFramework):
    def __init__(self):
        pass

    def from_benchmark(self, array):
        if array.data["format"] == "dense":
            data =  sp.asarray(array.data["values"]).reshape(array.data["shape"])
            return data
        if array.data["format"] == "COO":
            indices = []
            idx_dim = 0
            while "indices_" + str(idx_dim) in array.data:
                indices.append(array.data["indices_" + str(idx_dim)])
                idx_dim += 1
            V = array.data["values"]
            shape = array.data["shape"]
            return sp.COO(tuple(indices), V, shape=shape, fill_value=0)
        raise ValueError("Unsupported format: " + array.data["format"])

    def to_benchmark(self, array):
        if array.__class__ == sp.COO:
            return BinsparseFormat.from_coo(array.coords, array.data, array.shape)
        elif array.__class__ == sp.SparseArray:
            dense_array = array.todense()
            return BinsparseFormat.from_numpy(dense_array)

    def lazy(self, array):
        return array

    def compute(self, array):
        return array

    def einsum(self, prgm, **kwargs):
        return einsum(sp, prgm, **kwargs)

    def __getattr__(self, name):
        return getattr(sp, name)
