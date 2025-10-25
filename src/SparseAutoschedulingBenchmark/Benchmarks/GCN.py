"""
Name: Graph Convolutional Network Inference
Author: Tarun Devi
Email: tdevi3@gatech.edu

Motivation:
Graphs are widely used for abstracting complex systems of interacting objects,
such as social networks (Easley et al., 2010), knowledge graphs (Nickel et al.,
2015), molecular graphs (Wu et al., 2018), and biological networks (Barabasi &
Oltvai, 2004). They are also used for modeling 3D objects (Simonovsky &
Komodakis, 2017), manifolds (Bronstein et al., 2017), and source code
(Allamanis et al., 2017). Machine learning (ML), especially deep learning on
graphs, is an emerging field (Hamilton et al., 2017b; Bronstein et al., 2017).

W. Hu et al., “Open Graph Benchmark: Datasets for Machine Learning on Graphs,”
arXiv, vol. 2005.00687, pp. 1–15, Feb. 2021, doi: 10.48550/arXiv.2005.00687.

Role of Sparsity:
To represent a graph, an adjacency matrix is used, which is inherently sparse.

Implementation Details:
Implementation is hand-written and based on code referenced in the repository
README (https://anonymous.4open.science/r/scorch/README.md).

Data Generation:
Data generators have not been implemented yet — random weights are used.

Generative AI:
No generative AI was used to construct the benchmark function itself. This
statement was written by hand; generative AI may have been used to help with
tests.
"""

from ..BinsparseFormat import BinsparseFormat

adjacency_bench: BinsparseFormat
"""
benchmark_gcn(
    xp,
    adjacency_bench,
    features_bench,
    weights1_bench,
    bias1_bench,
    weights2_bench,
    bias2_bench,
)

Computes a 2-layer Graph Convolutional Network forward pass:

    h1 = ReLU(adjacency @ features @ weights1 + bias1)
    output = adjacency @ h1 @ weights2 + bias2

Args
----
xp : array_api
    Array API module (e.g. numpy, cupy, torch)
adjacency_bench : BinsparseFormat
    Sparse adjacency matrix of the graph
features_bench : BinsparseFormat
    Node feature matrix
weights1_bench : BinsparseFormat
    Weights for first GCN layer
bias1_bench : BinsparseFormat
    Bias for first GCN layer
weights2_bench : BinsparseFormat
    Weights for second GCN layer
bias2_bench : BinsparseFormat
    Bias for second GCN layer

Returns
-------
BinsparseFormat
    Output node embeddings after 2-layer GCN
"""


def benchmark_gcn(
    xp,
    adjacency_bench,
    features_bench,
    weights1_bench,
    bias1_bench,
    weights2_bench,
    bias2_bench,
):
    adjacency_lazy = xp.lazy(xp.from_benchmark(adjacency_bench))
    features_lazy = xp.lazy(xp.from_benchmark(features_bench))
    weights1_lazy = xp.lazy(xp.from_benchmark(weights1_bench))
    bias1_lazy = xp.lazy(xp.from_benchmark(bias1_bench))
    weights2_lazy = xp.lazy(xp.from_benchmark(weights2_bench))
    bias2_lazy = xp.lazy(xp.from_benchmark(bias2_bench))

    # Layer 1: adjacency @ features -> linear transform -> ReLU
    h1_lazy = adjacency_lazy @ features_lazy
    h1_lazy = h1_lazy @ weights1_lazy + bias1_lazy
    h1_lazy = xp.maximum(h1_lazy, 0)  # ReLU activation

    # Layer 2: adjacency @ h1 -> linear transform
    h2_lazy = adjacency_lazy @ h1_lazy
    output_lazy = h2_lazy @ weights2_lazy + bias2_lazy

    output_eager = xp.compute(output_lazy)
    return xp.to_benchmark(output_eager)
