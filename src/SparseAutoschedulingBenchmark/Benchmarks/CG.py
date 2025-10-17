def benchmark_cg(
    xp, A_bench, b_bench, x_bench, rel_tol=1e-8, abs_tol=1e-20, max_iters=10_000
):
    A = xp.lazy(xp.from_benchmark(A_bench))
    b = xp.lazy(xp.from_benchmark(b_bench))
    x = xp.lazy(xp.from_benchmark(x_bench))

    tolerance = max(xp.compute(xp.lazy(rel_tol) * norm(xp, b))[()], abs_tol)
    tol_sq = tolerance * tolerance
    max_iters = min(max_iters, xp.shape(A)[0])

    r = b - A @ x
    p = r
    it = 0
    rr = xp.vecdot(r, r)

    while xp.compute(rr)[()] >= tol_sq and it < max_iters:
        Ap = A @ p
        alpha = rr / xp.vecdot(p, Ap)
        x += alpha * p
        r -= alpha * Ap

        new_rr = xp.vecdot(r, r)

        if xp.compute(new_rr)[()] <= tol_sq:
            break

        beta = new_rr / rr
        p = r + beta * p
        rr = new_rr

        it += 1
    x_solution = xp.compute(x)
    return xp.to_benchmark(x_solution)


def norm(xp, v):
    return xp.sqrt(xp.sum(xp.multiply(v, v)))
