import numpy as np

from ..BinsparseFormat import BinsparseFormat
from .AbstractFramework import AbstractFramework


class LazyCheckerTensor:
    def __init__(self, array):
        if isinstance(array, np.ndarray):
            self.array = array
        else:
            self.array = array.array


class EagerCheckerTensor:
    def __init__(self, array):
        if isinstance(array, np.ndarray):
            self.array = array
        else:
            self.array = array.array


class CheckerOperator:
    def __init__(self, operator):
        self.operator = operator

    def __call__(self, *args, **kwds):
        for arg in args:
            if isinstance(arg, EagerCheckerTensor):
                raise AssertionError(
                    "Eager Tensors should always be made lazy before being operated on!"
                )
        arrays = [arg.array for arg in args]
        return LazyCheckerTensor(self.operator(*arrays, **kwds))


class CheckerFramework(AbstractFramework):
    """
    This framework uses numpy to perform operations, but checks that lazy and compute
    are used correctly in a benchmark function.
    """

    def __init__(self):
        pass

    def from_benchmark(self, array):
        if array.data["format"] == "dense":
            return EagerCheckerTensor(
                np.array(array.data["values"]).reshape(array.data["shape"])
            )
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
            return EagerCheckerTensor(data)
        raise ValueError("Unsupported format: " + array.data["format"])

    def to_benchmark(self, array):
        if isinstance(array, LazyCheckerTensor):
            raise AssertionError(
                "Lazy Tensors should always be computed before being converted to"
                " benchmark format!"
            )
        return BinsparseFormat.from_numpy(array.array)

    def lazy(self, array):
        return LazyCheckerTensor(array)

    def compute(self, array):
        return EagerCheckerTensor(array)

    def __getattr__(self, name):
        return CheckerOperator(getattr(np, name))
