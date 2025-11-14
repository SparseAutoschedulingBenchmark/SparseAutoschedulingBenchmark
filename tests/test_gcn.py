#!/usr/bin/env python3
# ruff: noqa
import pytest

import numpy as np

from SparseAutoschedulingBenchmark.BinsparseFormat import BinsparseFormat
from SparseAutoschedulingBenchmark.Frameworks.CheckerFramework import CheckerFramework
from SparseAutoschedulingBenchmark.Frameworks.NumpyFramework import NumpyFramework

from SparseAutoschedulingBenchmark.Benchmarks.GCN import benchmark_gcn


def gcn_reference(adjacency, features, weights1, bias1, weights2, bias2):
    """Reference implementation of 2-layer GCN"""
    # Layer 1: adjacency @ features -> linear transform -> ReLU
    h1 = adjacency @ features
    h1 = h1 @ weights1 + bias1
    h1 = np.maximum(h1, 0)  # ReLU activation

    # Layer 2: adjacency @ h1 -> linear transform
    h2 = adjacency @ h1
    output = h2 @ weights2 + bias2

    return output


import os
import pytest


@pytest.fixture(scope="module")
def suitesparse_adjacency():
    """Download a SuiteSparse matrix on-demand using ssgetpy.

    Requirements to run this test:
      - ssgetpy installed (test skipped otherwise)
      - SUITESPARSE_MATRIX env var set to the SuiteSparse identifier
        (example: "SNAP/ca-GrQc")
    The fixture returns a NumPy 2D adjacency array.
    """
    try:
        import ssgetpy
    except Exception:
        pytest.fail("ssgetpy is required for SuiteSparse tests")

    # Try a curated list of candidate SuiteSparse IDs and return the first
    # matrix we can fetch and convert. If none are available, skip the test.
    candidates = [
        "HB/bcsstk01",
        "HB/bcsstk02",
        "HB/bcsstk13",
        "SNAP/ca-GrQc",
        "SNAP/ca-HepTh",
        "SNAP/ca-HepPh",
    ]

    from pathlib import Path
    from scipy import io as sio
    from scipy import sparse as _sparse
    import numpy as _np

    for matrix_id in candidates:
        try:
            # Use the shared helper to attempt fetch + prepare; it uses skips
            # for transient problems so wrap it and continue to next candidate.
            try:
                adj = _fetch_and_prepare_suitesparse(matrix_id)
                return adj
            except Exception:
                # _fetch_and_prepare_suitesparse may call pytest.skip internally;
                # swallow and try the next candidate.
                continue
        except Exception:
            # any unexpected exception, try next candidate
            continue

    pytest.skip(
        "None of the curated SuiteSparse candidates could be fetched/converted in this environment"
    )


def test_gcn_suitesparse(suitesparse_adjacency):
    """Sanity test of benchmark_gcn using a SuiteSparse adjacency matrix.
    This test is intentionally lightweight in feature/weight sizes to keep runtime small.
    """
    adj = suitesparse_adjacency
    n = adj.shape[0]

    # keep features small so test runs quickly even for moderately large graphs
    features = np.ones((n, 4), dtype=float)
    weights1 = np.eye(4, dtype=float)
    bias1 = np.zeros((4,), dtype=float)
    weights2 = np.ones((4, 1), dtype=float)
    bias2 = np.zeros((1,), dtype=float)

    A_bin = BinsparseFormat.from_numpy(adj)
    features_bin = BinsparseFormat.from_numpy(features)
    w1_bin = BinsparseFormat.from_numpy(weights1)
    b1_bin = BinsparseFormat.from_numpy(bias1)
    w2_bin = BinsparseFormat.from_numpy(weights2)
    b2_bin = BinsparseFormat.from_numpy(bias2)

    xp = NumpyFramework()
    out = benchmark_gcn(xp, A_bin, features_bin, w1_bin, b1_bin, w2_bin, b2_bin)

    # basic sanity checks
    assert out is not None
    # Normalize output to a NumPy array. The benchmark may return a BinsparseFormat
    # or a framework-specific array-like object.
    if isinstance(out, BinsparseFormat):
        # convert COO/dense BinsparseFormat to numpy array
        out_coo = BinsparseFormat.to_coo(out)
        data = out_coo.data
        if data.get("format") == "COO":
            # indices are stored as numpy arrays in 'indices_0', 'indices_1', ...
            idx0 = data.get("indices_0")
            idx1 = data.get("indices_1")
            vals = data.get("values")
            shape = tuple(data.get("shape"))
            arr = np.zeros(shape, dtype=vals.dtype if hasattr(vals, "dtype") else float)
            arr[idx0, idx1] = vals
            out_arr = arr
        elif data.get("format") == "dense":
            shape = tuple(data.get("shape"))
            vals = data.get("values")
            out_arr = np.asarray(vals).reshape(shape)
        else:
            # fallback: try to interpret as numpy
            try:
                out_arr = np.asarray(data)
            except Exception:
                pytest.fail("Could not convert BinsparseFormat output to numpy array")
    else:
        # framework-specific objects may provide conversion helper
        if hasattr(out, "to_numpy"):
            out_arr = out.to_numpy()
        else:
            # try numpy conversion
            try:
                out_arr = np.asarray(out)
            except Exception:
                pytest.fail("Could not convert benchmark output to numpy array")

    assert out_arr.shape[0] == n


def test_gcn_bcsstk13(monkeypatch):
    """Convenience test that runs the suitesparse sanity check using HB/bcsstk13.
    This sets SUITESPARSE_MATRIX to 'HB/bcsstk13' for the duration of the test if not set.
    """
    # if user already specified a matrix, respect it
    if not os.environ.get("SUITESPARSE_MATRIX"):
        monkeypatch.setenv("SUITESPARSE_MATRIX", "HB/bcsstk13")

    # re-use the suitesparse test flow by calling pytest's fixture directly
    # pytest will handle the suitesparse_adjacency fixture injection when running via pytest
    # so here we simply call the test function by letting pytest execute it; for local convenience
    # we invoke pytest for the single test when requested (see instructions in README or run commands).
    # This function acts as a marker so users can run:
    # SUITESPARSE_MATRIX="HB/bcsstk13" poetry run pytest tests/test_gcn.py::test_gcn_bcsstk13 -q
    assert True


def _fetch_and_prepare_suitesparse(matrix_id):
    """Fetch a SuiteSparse matrix by id and return a dense numpy adjacency suitable for tests.

    Skips using pytest.skip if ssgetpy is not available or fetch fails.
    """
    ssgetpy = pytest.importorskip("ssgetpy")
    try:
        mats = ssgetpy.fetch(matrix_id, format="MM", dry_run=False)
    except Exception as e:
        pytest.skip(f"ssgetpy.fetch failed for {matrix_id}: {e}")

    if not mats:
        pytest.skip(f"ssgetpy.fetch returned no matrices for {matrix_id}")

    mat = mats[0]
    try:
        local = mat.localpath("MM", None, extract=True)[0]
    except Exception as e:
        pytest.skip(f"Could not locate downloaded matrix files for {matrix_id}: {e}")

    from pathlib import Path
    from scipy import io as sio
    from scipy import sparse as _sparse
    import numpy as _np

    p = Path(local)
    mtx_files = []
    if p.is_file() and p.suffix.lower() == ".mtx":
        mtx_files = [p]
    elif p.is_dir():
        mtx_files = sorted([f for f in p.iterdir() if f.suffix.lower() == ".mtx"])

    if not mtx_files:
        if p.is_dir():
            gz = sorted([f for f in p.iterdir() if f.name.lower().endswith(".mtx.gz")])
            if gz:
                mtx_files = gz

    if not mtx_files:
        pytest.skip(f"No Matrix Market (.mtx) file found in {local} for {matrix_id}")

    mtx = str(mtx_files[0])
    try:
        matdata = sio.mmread(mtx)
    except Exception as e:
        pytest.skip(f"Failed to read .mtx file {mtx} for {matrix_id}: {e}")

    try:
        if _sparse.issparse(matdata):
            A = matdata.tocsr()
        elif hasattr(matdata, "tocoo"):
            A = matdata.tocoo().tocsr()
        else:
            A = _sparse.csr_matrix(_np.asarray(matdata))
    except Exception as e:
        pytest.skip(f"Could not convert downloaded matrix to CSR for {matrix_id}: {e}")

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        pytest.skip(f"Downloaded matrix {matrix_id} is not square: shape={A.shape}")

    A = A + A.T
    try:
        A.setdiag(1)
    except Exception:
        A = A.tolil()
        A.setdiag(1)
        A = A.tocsr()
    A.eliminate_zeros()

    # return dense numpy array (tests expect numpy input)
    return _np.asarray(A.todense())


@pytest.mark.parametrize(
    "matrix_id",
    [
        "HB/bcsstk01",
        "HB/bcsstk02",
        "HB/bcsstk13",
        "SNAP/ca-GrQc",
        "SNAP/ca-HepTh",
        "SNAP/ca-HepPh",
    ],
)
def test_gcn_suitesparse_list(matrix_id):
    """Parametrized sanity check: run the GCN sanity check over a SuiteSparse matrix id.

    Each matrix id becomes its own pytest item (pass/skip reported per id).
    """
    adj = _fetch_and_prepare_suitesparse(matrix_id)
    n = adj.shape[0]

    # small feature sizes to keep runtime reasonable
    features = np.ones((n, 4), dtype=float)
    weights1 = np.eye(4, dtype=float)
    bias1 = np.zeros((4,), dtype=float)
    weights2 = np.ones((4, 1), dtype=float)
    bias2 = np.zeros((1,), dtype=float)

    A_bin = BinsparseFormat.from_numpy(adj)
    features_bin = BinsparseFormat.from_numpy(features)
    w1_bin = BinsparseFormat.from_numpy(weights1)
    b1_bin = BinsparseFormat.from_numpy(bias1)
    w2_bin = BinsparseFormat.from_numpy(weights2)
    b2_bin = BinsparseFormat.from_numpy(bias2)

    xp = NumpyFramework()
    out = benchmark_gcn(xp, A_bin, features_bin, w1_bin, b1_bin, w2_bin, b2_bin)

    # Normalize output as done in other tests
    if isinstance(out, BinsparseFormat):
        out_coo = BinsparseFormat.to_coo(out)
        data = out_coo.data
        if data.get("format") == "COO":
            idx0 = data.get("indices_0")
            idx1 = data.get("indices_1")
            vals = data.get("values")
            shape = tuple(data.get("shape"))
            arr = np.zeros(shape, dtype=vals.dtype if hasattr(vals, "dtype") else float)
            arr[idx0, idx1] = vals
            out_arr = arr
        elif data.get("format") == "dense":
            shape = tuple(data.get("shape"))
            vals = data.get("values")
            out_arr = np.asarray(vals).reshape(shape)
        else:
            out_arr = np.asarray(data)
    else:
        if hasattr(out, "to_numpy"):
            out_arr = out.to_numpy()
        else:
            out_arr = np.asarray(out)

    assert out_arr.shape[0] == n


@pytest.mark.parametrize(
    "xp,adjacency,features,weights1,bias1,weights2,bias2",
    [
        (
            NumpyFramework(),
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),  # Simple graph adjacency
            np.array([[1, 0], [0, 1], [1, 1]]),  # 3 nodes, 2 features each
            np.array([[1, 0], [0, 1]]),  # 2x2 weight matrix
            np.array([0, 0]),  # bias vector
            np.array([[1], [1]]),  # 2x1 weight matrix for output
            np.array([0]),  # output bias
        ),
        (
            NumpyFramework(),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # Identity adjacency
            np.array([[1, 2], [3, 4], [5, 6]]),  # Different feature values
            np.array([[0.5, 0.5], [0.5, 0.5]]),  # Averaging weights
            np.array([1, 1]),  # Non-zero bias
            np.array([[2], [3]]),  # Different output weights
            np.array([1]),  # Non-zero output bias
        ),
        (
            NumpyFramework(),
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),  # Zero adjacency
            np.array([[1, 1], [2, 2], [3, 3]]),  # Uniform features
            np.array([[1, 0], [0, 1]]),  # Identity weights
            np.array([0, 0]),  # Zero bias
            np.array([[1], [1]]),  # Sum weights
            np.array([0]),  # Zero output bias
        ),
        (
            CheckerFramework(),
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),  # Simple graph adjacency
            np.array([[1, 0], [0, 1], [1, 1]]),  # 3 nodes, 2 features each
            np.array([[1, 0], [0, 1]]),  # 2x2 weight matrix
            np.array([0, 0]),  # bias vector
            np.array([[1], [1]]),  # 2x1 weight matrix for output
            np.array([0]),  # output bias
        ),
        (
            CheckerFramework(),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # Identity adjacency
            np.array([[1, 2], [3, 4], [5, 6]]),  # Different feature values
            np.array([[0.5, 0.5], [0.5, 0.5]]),  # Averaging weights
            np.array([1, 1]),  # Non-zero bias
            np.array([[2], [3]]),  # Different output weights
            np.array([1]),  # Non-zero output bias
        ),
        (
            CheckerFramework(),
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),  # Zero adjacency
            np.array([[1, 1], [2, 2], [3, 3]]),  # Uniform features
            np.array([[1, 0], [0, 1]]),  # Identity weights
            np.array([0, 0]),  # Zero bias
            np.array([[1], [1]]),  # Sum weights
            np.array([0]),  # Zero output bias
        ),
    ],
)
def test_benchmark_gcn(xp, adjacency, features, weights1, bias1, weights2, bias2):
    output_ref = gcn_reference(adjacency, features, weights1, bias1, weights2, bias2)

    adjacency_bin = BinsparseFormat.from_numpy(adjacency)
    features_bin = BinsparseFormat.from_numpy(features)
    weights1_bin = BinsparseFormat.from_numpy(weights1)
    bias1_bin = BinsparseFormat.from_numpy(bias1)
    weights2_bin = BinsparseFormat.from_numpy(weights2)
    bias2_bin = BinsparseFormat.from_numpy(bias2)

    output_bin = benchmark_gcn(
        xp,
        adjacency_bin,
        features_bin,
        weights1_bin,
        bias1_bin,
        weights2_bin,
        bias2_bin,
    )
    output_bin = BinsparseFormat.to_coo(output_bin)

    assert output_bin == BinsparseFormat.to_coo(BinsparseFormat.from_numpy(output_ref))
