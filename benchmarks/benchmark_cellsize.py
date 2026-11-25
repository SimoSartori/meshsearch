"""What the cell side costs: build time, memory, cell occupancy and the three
query kinds, swept over cell sides of one to four mean separations (in half
steps up to three) at two data sizes.

The cell side is meshsearch's one tuning parameter. A fine grid holds few
objects per cell and many cells; a coarse one the reverse, and every cell
costs memory (its block, and eight entries per cell in the shared offset
table), so the cell count and the memory with it fall as the cube of the
factor. The radius is held at twice the mean separation so that the answer
stays the same size at every cell side (about 33 neighbours on uniform data).

The tables on the documentation's performance page, docs/benchmark.md, were
produced with this script.

    python benchmarks/benchmark_cellsize.py --sweep          # every size, cell side and input
    python benchmarks/benchmark_cellsize.py --sweep --dists clustered
    python benchmarks/benchmark_cellsize.py --one-sweep 1000000 3 --dist clustered   # one, as JSON
    python benchmarks/benchmark_cellsize.py --sweep --queries repeat   # the same 1000 queries each time

Each query kind is timed per call and batched (all the queries of a
repetition in one call); a per call time includes the cost of the Python call
itself, which a batch does not pay, so a batched time is the one that shows
what the cell side does. Times are the median of three repetitions.

--queries fresh (the default) draws 10,000 new query points for every
repetition, as benchmark.py does, so that no repetition finds the cells it
visits in cache from an earlier one. --queries repeat runs the same 1000 query
points in every repetition, which measures a working set already in cache and
favours the finer cell sides.

The inputs are the ones of benchmark.py: "uniform" points in the unit cube, or
"clustered", a Thomas process described there. Queries come from the same
distribution as the data.
"""

import argparse
import gc
import json
import subprocess
import sys
import time

import numpy as np

from benchmark import DISTS, N_QUERIES, columns, describe, points, query_sets

REPEATS = 3
K = 10
SEED = 20260921
FACTORS = [1, 1.5, 2, 2.5, 3, 4]
SIZES = [1_000_000, 10_000_000]


def rss_mib():
    import psutil
    return psutil.Process().memory_info().rss / 2 ** 20


def timed(fn, sets, n_queries):
    """Median wall time of fn over the query sets, one run per set, in microseconds per query."""
    samples = []
    for q in sets:
        t = time.perf_counter()
        fn(q)
        samples.append(time.perf_counter() - t)
    return float(np.median(samples)) / n_queries * 1e6


def measure(grid, sets, radius):
    csets = [columns(q) for q in sets]
    nq = len(sets[0])
    return {
        "nn_batch_us": timed(lambda c: grid.nearest_object(*c), csets, nq),
        "knn_batch_us": timed(lambda c: grid.nearest_objects(K, *c), csets, nq),
        "radius_batch_us": timed(lambda c: grid.close_objects(*c, radius), csets, nq),
        "nn_us": timed(lambda q: [grid.nearest_object(*p) for p in q], sets, nq),
        "knn_us": timed(lambda q: [grid.nearest_objects(K, *p) for p in q], sets, nq),
        "radius_us": timed(lambda q: [grid.close_objects(*p, radius) for p in q], sets, nq),
        "radius_count": float(np.mean(grid.count_objects(*csets[0], radius))),
    }


def occupancy(grid, x, y, z):
    """How the objects spread over the cells: the fraction of cells left
    empty, the mean count over the cells that are not, and the 99th
    percentile of the count over all cells. Cells are located from the
    grid's own box and cell side, with the last cell of an axis taking a
    point on its upper edge."""
    lims, side, shape = grid.lims, grid.cellsize, grid.n_cells
    linear = np.zeros(len(x), dtype=np.int64)
    for axis, coord in enumerate((x, y, z)):
        cell = np.floor((coord - lims[axis, 0]) / side).astype(np.int64)
        linear = linear * shape[axis] + np.clip(cell, 0, shape[axis] - 1)
    counts = np.bincount(linear, minlength=shape[0] * shape[1] * shape[2])
    filled = counts[counts > 0]
    return {"empty_fraction": float(1 - filled.size / counts.size),
            "mean_per_filled_cell": float(filled.mean()),
            "p99_per_cell": float(np.percentile(counts, 99))}


def run_sweep(n, factor, dist="uniform", mode="fresh"):
    import meshsearch
    x, y, z = columns(points(n, dist))
    sets = query_sets(N_QUERIES[mode], n, dist, mode, SEED + 1, REPEATS)
    mps = n ** (-1 / 3)
    cellsize = factor * mps
    radius = 2 * mps                      # the answer size is held constant

    gc.collect()
    before = rss_mib()
    t = time.perf_counter()
    grid = meshsearch.MeshGrid(x, y, z, cellsize, limits=[[0, 1]] * 3)
    build = time.perf_counter() - t
    memory = rss_mib() - before

    cells = grid.n_cells[0] * grid.n_cells[1] * grid.n_cells[2]
    result = {"dist": dist, "n": n, "factor": factor, "cellsize": cellsize, "build_s": build,
              "memory_mib": memory, "n_cells": list(grid.n_cells), "cells": cells,
              "objects_per_cell": n / cells, "n_queries": N_QUERIES[mode], "queries": mode}
    result.update(occupancy(grid, x, y, z))
    result.update(describe(dist, n))
    result.update(measure(grid, sets, radius))
    return result


def spawn(args_list):
    proc = subprocess.run([sys.executable, __file__] + args_list,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"failed: {args_list}")
    return json.loads(proc.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--one-sweep", nargs=2, type=float, metavar=("N", "FACTOR"))
    parser.add_argument("--dist", choices=DISTS, default="uniform", help="with --one-sweep")
    parser.add_argument("--dists", nargs="*", choices=DISTS, default=DISTS)
    parser.add_argument("--queries", choices=["fresh", "repeat"], default="fresh",
                        help="new query points for every repetition (default), or the same ones")
    args = parser.parse_args()

    if args.one_sweep:
        n, factor = args.one_sweep
        print(json.dumps(run_sweep(int(n), factor, args.dist, args.queries)))
    elif args.sweep:
        rows = []
        for dist in args.dists:
            for n in SIZES:
                for factor in FACTORS:
                    sys.stderr.write(f"  sweep {dist} n={n:,} cellsize={factor} mps ... ")
                    sys.stderr.flush()
                    row = spawn(["--one-sweep", str(n), str(factor), "--dist", dist,
                                 "--queries", args.queries])
                    sys.stderr.write(f"{row['memory_mib']:,.0f} MiB\n")
                    rows.append(row)
        print(json.dumps(rows, indent=2))
    else:
        parser.error("choose --sweep or --one-sweep N FACTOR")


if __name__ == "__main__":
    main()
