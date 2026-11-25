"""Benchmark meshsearch against scipy's cKDTree and sklearn's BallTree.

Each library/size combination runs in its own process, so the memory figure is
the resident-set growth caused by holding that one structure.

    python benchmarks/benchmark.py                      # uniform and clustered, cell sides 1 and 4
    python benchmarks/benchmark.py --dists clustered --factors 4
    python benchmarks/benchmark.py --periodic --sizes 1000000 10000000
    python benchmarks/benchmark.py --large-batch --sizes 1000000 10000000
    python benchmarks/benchmark.py --one meshsearch 100000 --factor 4 --dist clustered   # one, as JSON
    python benchmarks/benchmark.py --queries repeat     # the same 1000 queries in every repetition

Nothing here is tuned to favour any library. meshsearch is given a cell side
that is a stated multiple of the mean separation, and every structure is built
with default parameters and queried single-threaded. The trees have no cell
side, so their rows are the same measurement whichever factor is passed.

Two kinds of input. "uniform" is points uniform in the unit cube. "clustered"
is a Thomas process in the unit cube: parent centres uniform, offspring
Gaussian around a parent, plus a uniform background, with the parameters below
chosen to give strong clustering. The queries are drawn from the same
distribution as the data: uniform ones for uniform data, fresh draws from the
same process (same parents) for clustered data. Everything is wrapped into the
unit cube, so the same points serve the periodic and the non-periodic runs.

--periodic runs meshsearch with all three axes periodic against
cKDTree(boxsize=1); BallTree has no periodic mode and is left out.
--large-batch times one call of 1e5 and 1e6 queries on uniform data, which is
where meshsearch runs a batch in cell order (it does from 1e5 queries).

--queries fresh (the default) draws new query points for every repetition,
10,000 per repetition (1e5 or 1e6 with --large-batch), the same draws for every
library, so that no repetition finds the cells or tree nodes it visits in cache
from an earlier one. --queries repeat runs the same 1000 query points (the
same large batch) in every repetition, which measures a working set that is
already in cache.
"""

import argparse
import gc
import json
import subprocess
import sys
import time

import numpy as np

SIZES = [100_000, 1_000_000, 10_000_000]
LIBRARIES = ["meshsearch", "cKDTree", "BallTree"]

N_QUERIES = {"fresh": 10_000, "repeat": 1000}
REPEATS = 3
K = 10
SEED = 20260921
LARGE_BATCHES = [100_000, 1_000_000]

# Thomas process, fixed for every size so that the structure scales with the
# mean separation: 90 per cent of the points in clusters of about 1000 points
# with a Gaussian scale of one mean separation, the rest uniform.
CLUSTER_FRACTION = 0.9
CLUSTER_POINTS = 1000
CLUSTER_SIGMA_MPS = 1.0
DISTS = ["uniform", "clustered"]


def n_clusters(n):
    return max(1, round(CLUSTER_FRACTION * n / CLUSTER_POINTS))


def thomas(n, m, seed):
    """m points of the Thomas process whose parents are those of an n-point data set."""
    parents = np.random.default_rng(SEED + 7).random((n_clusters(n), 3))
    rng = np.random.default_rng(seed)
    m_cluster = round(CLUSTER_FRACTION * m)
    sigma = CLUSTER_SIGMA_MPS * n ** (-1 / 3)
    centre = parents[rng.integers(len(parents), size=m_cluster)]
    pos = np.concatenate([centre + rng.normal(0.0, sigma, (m_cluster, 3)),
                          rng.random((m - m_cluster, 3))])
    pos = pos[rng.permutation(m)]
    pos = np.mod(pos, 1.0)
    pos[pos >= 1.0] = 0.0
    return pos


def points(n, dist="uniform"):
    if dist == "uniform":
        return np.random.default_rng(SEED).random((n, 3))
    return thomas(n, n, SEED)


def query_points(n_queries, n=0, dist="uniform", seed=SEED + 1):
    if dist == "uniform":
        return np.random.default_rng(seed).random((n_queries, 3))
    return thomas(n, n_queries, seed)


def query_sets(n_queries, n, dist, mode, first_seed, reps=REPEATS):
    """One set of query points per repetition: new ones each time if fresh, else one set repeated."""
    if mode == "fresh":
        return [query_points(n_queries, n, dist, first_seed + 1000 * r) for r in range(reps)]
    return [query_points(n_queries, n, dist, first_seed)] * reps


def columns(a):
    return [np.ascontiguousarray(a[:, i]) for i in range(3)]


def describe(dist, n):
    if dist == "uniform":
        return {"dist": dist}
    return {"dist": dist, "n_clusters": n_clusters(n), "cluster_fraction": CLUSTER_FRACTION,
            "cluster_sigma_mps": CLUSTER_SIGMA_MPS, "points_per_cluster": CLUSTER_POINTS}


def rss_mib():
    import psutil
    return psutil.Process().memory_info().rss / 2 ** 20


def timed(fn, sets):
    """Median wall time of fn over the query sets, one run per set, in seconds."""
    samples = []
    for q in sets:
        t = time.perf_counter()
        fn(q)
        samples.append(time.perf_counter() - t)
    return float(np.median(samples))


def run_one(library, n, factor=1, dist="uniform", periodic=False, mode="fresh"):
    data = points(n, dist)
    sets = query_sets(N_QUERIES[mode], n, dist, mode, SEED + 1)
    csets = [columns(q) for q in sets]
    mps = n ** (-1 / 3)
    radius = 2 * mps
    cellsize = factor * mps
    x, y, z = columns(data)

    gc.collect()
    before = rss_mib()

    result = {"library": library, "n": n, "k": K, "radius": radius, "factor": factor,
              "n_queries": N_QUERIES[mode], "queries": mode, "periodic": periodic}
    result.update(describe(dist, n))

    if library == "meshsearch":
        import meshsearch
        t = time.perf_counter()
        grid = meshsearch.MeshGrid(x, y, z, cellsize, limits=[[0, 1]] * 3, periodic=periodic)
        result["build_s"] = time.perf_counter() - t
        result["memory_mib"] = rss_mib() - before
        result["n_cells"] = list(grid.n_cells)
        result["cellsize_used"] = grid.cellsize

        result["nn_loop_s"] = timed(
            lambda q: [grid.nearest_object(*p) for p in q], sets)
        result["knn_loop_s"] = timed(
            lambda q: [grid.nearest_objects(K, *p) for p in q], sets)
        result["radius_loop_s"] = timed(
            lambda q: [grid.close_objects(*p, radius) for p in q], sets)

        # the batched forms, which the bindings add and the C++ library does not
        result["nn_batch_s"] = timed(lambda c: grid.nearest_object(*c), csets)
        result["knn_batch_s"] = timed(lambda c: grid.nearest_objects(K, *c), csets)
        result["radius_batch_s"] = timed(lambda c: grid.close_objects(*c, radius), csets)
        result["radius_count"] = float(np.mean(grid.count_objects(*csets[0], radius)))
        del grid

    elif library == "cKDTree":
        from scipy.spatial import cKDTree
        boxsize = 1.0 if periodic else None
        t = time.perf_counter()
        tree = cKDTree(data, boxsize=boxsize)
        result["build_s"] = time.perf_counter() - t
        result["memory_mib"] = rss_mib() - before

        result["nn_batch_s"] = timed(lambda q: tree.query(q, k=1), sets)
        result["knn_batch_s"] = timed(lambda q: tree.query(q, k=K), sets)
        result["radius_batch_s"] = timed(
            lambda q: tree.query_ball_point(q, radius), sets)
        result["nn_loop_s"] = timed(
            lambda q: [tree.query(p, k=1) for p in q], sets)
        result["knn_loop_s"] = timed(
            lambda q: [tree.query(p, k=K) for p in q], sets)
        result["radius_loop_s"] = timed(
            lambda q: [tree.query_ball_point(p, radius) for p in q], sets)
        result["radius_count"] = float(np.mean(
            tree.query_ball_point(sets[0], radius, return_length=True)))
        del tree

    elif library == "BallTree":
        if periodic:
            raise SystemExit("BallTree has no periodic mode")
        from sklearn.neighbors import BallTree
        t = time.perf_counter()
        tree = BallTree(data)
        result["build_s"] = time.perf_counter() - t
        result["memory_mib"] = rss_mib() - before

        result["nn_batch_s"] = timed(lambda q: tree.query(q, k=1), sets)
        result["knn_batch_s"] = timed(lambda q: tree.query(q, k=K), sets)
        result["radius_batch_s"] = timed(
            lambda q: tree.query_radius(q, radius), sets)
        result["nn_loop_s"] = timed(
            lambda q: [tree.query(p.reshape(1, -1), k=1) for p in q], sets)
        result["knn_loop_s"] = timed(
            lambda q: [tree.query(p.reshape(1, -1), k=K) for p in q], sets)
        result["radius_loop_s"] = timed(
            lambda q: [tree.query_radius(p.reshape(1, -1), radius) for p in q], sets)
        result["radius_count"] = float(np.mean(
            tree.query_radius(sets[0], radius, count_only=True)))
        del tree

    else:
        raise SystemExit(f"unknown library {library}")

    return result


def run_large(library, n, factor=1, batches=LARGE_BATCHES, reps=REPEATS, mode="fresh"):
    """One call of each query kind over a large batch, on uniform data.

    For meshsearch the batch is also cut into chunks of 50,000 queries, below
    the 1e5 at which it switches to cell order, so that the same queries run in
    the order they were given; the ratio of the two is what cell order buys.
    """
    data = points(n)
    mps = n ** (-1 / 3)
    radius = 2 * mps
    x, y, z = columns(data)
    gc.collect()
    result = {"library": library, "n": n, "factor": factor, "k": K, "radius": radius,
              "reps": reps, "queries": mode, "dist": "uniform", "batches": {}}

    if library == "meshsearch":
        import meshsearch
        grid = meshsearch.MeshGrid(x, y, z, factor * mps, limits=[[0, 1]] * 3)
        calls = {"nn": lambda a, b, c: grid.nearest_object(a, b, c),
                 "knn": lambda a, b, c: grid.nearest_objects(K, a, b, c),
                 "radius": lambda a, b, c: grid.close_objects(a, b, c, radius)}
    elif library == "cKDTree":
        from scipy.spatial import cKDTree
        tree = cKDTree(data)
        calls = {"nn": lambda q: tree.query(q, k=1),
                 "knn": lambda q: tree.query(q, k=K),
                 "radius": lambda q: tree.query_ball_point(q, radius)}
    else:
        raise SystemExit(f"unknown library {library}")

    chunk = 50_000
    for nq in batches:
        sets = query_sets(nq, n, "uniform", mode, SEED + 2, reps)
        csets = [columns(q) for q in sets]
        row = {}
        for kind, call in calls.items():
            if library == "meshsearch":
                row[f"{kind}_batch_s"] = timed(lambda c: call(*c), csets)
            else:
                row[f"{kind}_batch_s"] = timed(lambda q: call(q), sets)
            if library == "meshsearch" and nq > chunk:
                def chunked(c):
                    qx, qy, qz = c
                    for lo in range(0, nq, chunk):
                        call(qx[lo:lo + chunk], qy[lo:lo + chunk], qz[lo:lo + chunk])
                row[f"{kind}_chunked_s"] = timed(chunked, csets)
        result["batches"][str(nq)] = row
    return result


def spawn(args_list):
    proc = subprocess.run([sys.executable, __file__] + args_list,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": proc.stderr.strip().splitlines()[-1:]}
    return json.loads(proc.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", nargs=2, metavar=("LIBRARY", "N"))
    parser.add_argument("--one-large", nargs=2, metavar=("LIBRARY", "N"))
    parser.add_argument("--sizes", type=int, nargs="*", default=SIZES)
    parser.add_argument("--factor", type=int, default=1,
                        help="cell side, in mean separations (meshsearch only), with --one")
    parser.add_argument("--factors", type=int, nargs="*", default=[1, 4],
                        help="cell sides to run, in mean separations")
    parser.add_argument("--dist", choices=DISTS, default="uniform", help="with --one")
    parser.add_argument("--dists", nargs="*", choices=DISTS, default=DISTS)
    parser.add_argument("--periodic", action="store_true",
                        help="meshsearch all-periodic against cKDTree(boxsize=1)")
    parser.add_argument("--large-batch", action="store_true")
    parser.add_argument("--reps", type=int, default=REPEATS, help="repetitions of the large batches")
    parser.add_argument("--queries", choices=["fresh", "repeat"], default="fresh",
                        help="new query points for every repetition (default), or the same ones")
    args = parser.parse_args()

    if args.one:
        print(json.dumps(run_one(args.one[0], int(args.one[1]), args.factor, args.dist,
                                 args.periodic, args.queries)))
        return
    if args.one_large:
        print(json.dumps(run_large(args.one_large[0], int(args.one_large[1]), args.factor,
                                   reps=args.reps, mode=args.queries)))
        return

    results = []
    libraries = ["meshsearch", "cKDTree"] if (args.periodic or args.large_batch) else LIBRARIES
    for dist in (["uniform"] if args.large_batch else args.dists):
        for n in args.sizes:
            for library in libraries:
                for factor in (args.factors if library == "meshsearch" else [0]):
                    label = f"{dist} {library} n={n:,} factor={factor}"
                    sys.stderr.write(f"  {label} ... ")
                    sys.stderr.flush()
                    t = time.perf_counter()
                    if args.large_batch:
                        row = spawn(["--one-large", library, str(n), "--factor", str(factor),
                                     "--reps", str(args.reps), "--queries", args.queries])
                    else:
                        row = spawn(["--one", library, str(n), "--factor", str(factor),
                                     "--dist", dist, "--queries", args.queries]
                                    + (["--periodic"] if args.periodic else []))
                    if "error" in row:
                        row.update({"library": library, "n": n, "dist": dist})
                    results.append(row)
                    sys.stderr.write(f"{time.perf_counter() - t:.0f} s\n")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
