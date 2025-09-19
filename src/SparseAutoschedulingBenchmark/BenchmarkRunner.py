import argparse
import time

from .Benchmarks.MatMul import (
    benchmark_matmul,
    dg_matmul_dense_large,
    dg_matmul_dense_small,
    dg_matmul_sparse_large,
    dg_matmul_sparse_small,
)
from .Frameworks.CheckerFramework import CheckerFramework
from .Frameworks.NumpyFramework import NumpyFramework

FRAMEWORK_DICT = {"numpy": NumpyFramework(), "checker": CheckerFramework()}
BENCHMARK_DICT = {"matmul": benchmark_matmul}
DATA_GENERATOR_DICT = {
    "matmul": {
        "dense_small": dg_matmul_dense_small,
        "dense_large": dg_matmul_dense_large,
        "sparse_small": dg_matmul_sparse_small,
        "sparse_large": dg_matmul_sparse_large,
    }
}


def run_benchmark(framework, benchmark_function, benchmark_data_generator, iters):
    execution_times = []
    for _ in range(iters):
        data = benchmark_data_generator()
        start = time.perf_counter()
        benchmark_function(framework, *data)
        end = time.perf_counter()
        duration = end - start
        execution_times.append(duration)
    print(
        f"Benchmark took an average of {sum(execution_times) / len(execution_times)}\
             seconds"
    )
    return execution_times


def save_benchmark_results(
    results_folder, execution_times, framework, benchmark, data_generator
):
    filename = f"{results_folder}/{framework}_{benchmark}_{data_generator}.csv"
    with open(filename, "w") as f:
        f.write("Framework,Benchmark,Data Generator,Iteration,ExecutionTime\n")
        for i, execution_time in enumerate(execution_times):
            f.write(f"{framework},{benchmark},{data_generator},{i},{execution_time}\n")


# This function allows either command line arguments or direct function calls to
# run benchmarks. Note that if frameworks, benchmarks, or data_generators are
# provided directly, the corresponding command line arguments are ignored. Further,
# if a user benchmark is provided, data generators must also be provided directly.
def main(
    frameworks=None,
    framework_names=None,
    benchmarks=None,
    benchmark_names=None,
    data_generators=None,
    data_generator_names=None,
    iters=None,
    results_folder=None,
):
    parser = argparse.ArgumentParser(description="Run sparse autoscheduling benchmark")
    parser.add_argument(
        "--framework",
        default=["all"],
        nargs="*",
        help="Execution framework(s) to use",
    )
    parser.add_argument(
        "--benchmark",
        default=["all"],
        nargs="*",
        help="Benchmark(s) to run",
    )
    parser.add_argument(
        "--data-generator",
        default=["all"],
        nargs="*",
        help="Data generator(s) to use",
    )
    parser.add_argument(
        "--iterations",
        default=5,
        type=int,
        help="Number of iterations to run for each benchmark",
    )
    parser.add_argument(
        "--results-folder", default="results", help="Folder to save results"
    )
    args = parser.parse_args()
    if frameworks is None:
        if args.framework == ["all"]:
            frameworks = list(FRAMEWORK_DICT.items())
        else:
            frameworks = [(fw, FRAMEWORK_DICT[fw]) for fw in args.framework]
    else:
        frameworks = zip(framework_names, frameworks, strict=False)

    if benchmarks is None:
        if args.benchmark == ["all"]:
            benchmarks = list(BENCHMARK_DICT.items())
        else:
            benchmarks = [
                (benchmark_name, BENCHMARK_DICT[benchmark_name])
                for benchmark_name in args.benchmark
            ]
    else:
        benchmarks = zip(benchmark_names, benchmarks, strict=False)

    user_submitted_dgs = data_generators is not None
    if not user_submitted_dgs:
        if args.data_generator == ["all"]:
            data_generators = []
            for benchmark_name, _bench in benchmarks:
                for dg in DATA_GENERATOR_DICT[benchmark_name]:
                    data_generators.append(
                        (dg, DATA_GENERATOR_DICT[benchmark_name][dg])
                    )
        else:
            data_generators = []
            for benchmark_name, _bench in benchmarks:
                for dg in args.data_generator:
                    if dg in DATA_GENERATOR_DICT[benchmark_name]:
                        data_generators.append(
                            (dg, DATA_GENERATOR_DICT[benchmark_name][dg])
                        )
    else:
        data_generators = zip(data_generator_names, data_generators, strict=False)

    if results_folder is None:
        results_folder = args.results_folder

    if iters is None:
        iters = args.iterations

    for framework_name, framework in frameworks:
        for benchmark_name, benchmark in benchmarks:
            for data_generator_name, data_generator in data_generators:
                if (
                    not user_submitted_dgs
                    and data_generator_name not in DATA_GENERATOR_DICT[benchmark_name]
                ):
                    continue
                print(
                    f"Running benchmark {benchmark_name} with framework\
                          {framework_name} and data generator {data_generator_name}"
                )
                execution_times = run_benchmark(
                    framework, benchmark, data_generator, iters
                )
                save_benchmark_results(
                    results_folder,
                    execution_times,
                    framework_name,
                    benchmark_name,
                    data_generator_name,
                )
