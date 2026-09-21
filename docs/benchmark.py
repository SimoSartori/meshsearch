"""Benchmark meshsearch against scipy's cKDTree and sklearn's BallTree.

Each library/size combination runs in its own process, so the memory figure is
the resident-set growth caused by holding that one structure.

    python docs/benchmark.py --factor 1   # cell side of one mean separation
    python docs/benchmark.py --factor 4   # four, the regime the library is used in
    python docs/benchmark.py --one meshsearch 100000 --factor 4   # one, as JSON

Nothing here is tuned to favour any library. meshsearch is given a cell side
that is a stated multiple of the mean separation, and every structure is built
with default parameters and queried single-threaded. The trees have no cell
side, so their rows are the same measurement whichever factor is passed.
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

N_QUERIES = 1000
K = 10
SEED = 20260921


def points(n):
    rng = np.random.default_rng(SEED)
    return rng.random((n, 3))


def query_points(n_queries):
    rng = np.random.default_rng(SEED + 1)
    return rng.random((n_queries, 3))


def rss_mib():
    import psutil
    return psutil.Process().memory_info().rss / 2 ** 20


def timed(fn, repeats):
    """Median wall time of `repeats` runs, in seconds."""
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    return float(np.median(samples))


def run_one(library, n, factor=1):
    data = points(n)
    queries = query_points(N_QUERIES)
    mps = n ** (-1 / 3)
    radius = 2 * mps
    cellsize = factor * mps

    gc.collect()
    before = rss_mib()

    result = {"library": library, "n": n, "k": K, "radius": radius,
              "factor": factor, "n_queries": N_QUERIES}

    if library == "meshsearch":
        import meshsearch
        x, y, z = np.ascontiguousarray(data[:, 0]), np.ascontiguousarray(data[:, 1]), \
            np.ascontiguousarray(data[:, 2])
        t = time.perf_counter()
        grid = meshsearch.MeshGrid(x, y, z, cellsize)
        result["build_s"] = time.perf_counter() - t
        result["memory_mib"] = rss_mib() - before

        result["nn_loop_s"] = timed(
            lambda: [grid.nearest_object(*p) for p in queries], 3)
        result["knn_loop_s"] = timed(
            lambda: [grid.nearest_objects(K, *p) for p in queries], 3)
        result["radius_loop_s"] = timed(
            lambda: [grid.close_objects(*p, radius) for p in queries], 3)

        # the batched forms, which the bindings add and the C++ library does not
        qx = np.ascontiguousarray(queries[:, 0])
        qy = np.ascontiguousarray(queries[:, 1])
        qz = np.ascontiguousarray(queries[:, 2])
        result["nn_batch_s"] = timed(lambda: grid.nearest_object(qx, qy, qz), 3)
        result["knn_batch_s"] = timed(lambda: grid.nearest_objects(K, qx, qy, qz), 3)
        result["radius_batch_s"] = timed(lambda: grid.close_objects(qx, qy, qz, radius), 3)
        result["radius_count"] = float(np.mean(
            [grid.close_objects(*p, radius).size for p in queries]))
        del grid

    elif library == "cKDTree":
        from scipy.spatial import cKDTree
        t = time.perf_counter()
        tree = cKDTree(data)
        result["build_s"] = time.perf_counter() - t
        result["memory_mib"] = rss_mib() - before

        result["nn_batch_s"] = timed(lambda: tree.query(queries, k=1), 3)
        result["knn_batch_s"] = timed(lambda: tree.query(queries, k=K), 3)
        result["radius_batch_s"] = timed(
            lambda: tree.query_ball_point(queries, radius), 3)
        result["nn_loop_s"] = timed(
            lambda: [tree.query(p, k=1) for p in queries], 3)
        result["knn_loop_s"] = timed(
            lambda: [tree.query(p, k=K) for p in queries], 3)
        result["radius_loop_s"] = timed(
            lambda: [tree.query_ball_point(p, radius) for p in queries], 3)
        result["radius_count"] = float(np.mean(
            [len(tree.query_ball_point(p, radius)) for p in queries]))
        del tree

    elif library == "BallTree":
        from sklearn.neighbors import BallTree
        t = time.perf_counter()
        tree = BallTree(data)
        result["build_s"] = time.perf_counter() - t
        result["memory_mib"] = rss_mib() - before

        result["nn_batch_s"] = timed(lambda: tree.query(queries, k=1), 3)
        result["knn_batch_s"] = timed(lambda: tree.query(queries, k=K), 3)
        result["radius_batch_s"] = timed(
            lambda: tree.query_radius(queries, radius), 3)
        result["nn_loop_s"] = timed(
            lambda: [tree.query(p.reshape(1, -1), k=1) for p in queries], 3)
        result["knn_loop_s"] = timed(
            lambda: [tree.query(p.reshape(1, -1), k=K) for p in queries], 3)
        result["radius_loop_s"] = timed(
            lambda: [tree.query_radius(p.reshape(1, -1), radius) for p in queries], 3)
        result["radius_count"] = float(np.mean(
            [tree.query_radius(p.reshape(1, -1), radius)[0].size for p in queries]))
        del tree

    else:
        raise SystemExit(f"unknown library {library}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", nargs=2, metavar=("LIBRARY", "N"))
    parser.add_argument("--sizes", type=int, nargs="*", default=SIZES)
    parser.add_argument("--factor", type=int, default=1,
                        help="cell side, in mean separations (meshsearch only)")
    args = parser.parse_args()

    if args.one:
        print(json.dumps(run_one(args.one[0], int(args.one[1]), args.factor)))
        return

    results = []
    for n in args.sizes:
        for library in LIBRARIES:
            sys.stderr.write(f"  {library} at {n:,} ... ")
            sys.stderr.flush()
            proc = subprocess.run(
                [sys.executable, __file__, "--one", library, str(n),
                 "--factor", str(args.factor)],
                capture_output=True, text=True)
            if proc.returncode != 0:
                sys.stderr.write("FAILED\n")
                results.append({"library": library, "n": n,
                                "error": proc.stderr.strip().splitlines()[-1:]})
                continue
            row = json.loads(proc.stdout)
            results.append(row)
            sys.stderr.write(f"build {row['build_s']:.2f}s\n")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
