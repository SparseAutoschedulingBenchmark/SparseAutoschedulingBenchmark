# SparseApplicationBenchmark

Sparse array programming frameworks, such as [SciPy](https://scipy.org) or [pydata/sparse](https://sparse.pydata.org/en/stable/), are getting more advanced. Because the performance and optimization strategies for sparse frameworks depends heavily on the input sparsity patterns and programs, we need realistic applications to make informed design decisions. This benchmark suite consists of several applications written entirely using collective operations (such as +, *, sum, or reduce) over sparse arrays. The programs are adapted from real-world applications using a straightforward translation to an [array-programming](https://en.wikipedia.org/wiki/Array_programming#Array_languages) style. The standard form for our benchmark functions is vanilla python code using only [Array-API]( https://data-apis.org/array-api/latest/API_specification/) functions, with minimal looping or control flow.

We take great inspiration from the great benchmarks in the database community, such as [pandasbench](https://arxiv.org/abs/2506.02345), [Join Order Benchmark](https://dl.acm.org/doi/10.14778/2850583.2850594), or [TPC-H](https://www.tpc.org/tpch/)

## Contributing
Start with the [policy doc](https://docs.google.com/document/d/1N5gElU3Z_URG-K4HTdLlf_H1lpI42dH9xVdeBs4ierA/edit?usp=sharing), which describes the policies and processes by which you can contribute benchmarks to the repo.

Most importantly: Before implementing a benchmark, claim it! Select or create your own github issue describing which application you want to benchmark, with links to relevant source code, and assign yourself to the issue if possible so that others know you're working on that benchmark. You can only claim a benchmark for a maximum of 1 month, after which others can claim it.

Once you're on board, see [CONTRIBUTING.md](CONTRIBUTING.md) for software guidelines, development setup, and best practices.

## Installation

SparseAutoschedulingBenchmark uses [poetry](https://python-poetry.org/) for packaging. To install for
development, clone the repository and run:
```bash
poetry install --with test
```
to install the current project and dev dependencies.

