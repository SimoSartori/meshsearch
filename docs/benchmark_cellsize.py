"""What the cell side costs: build time, memory and the three query kinds,
swept over cell sides of one to four mean separations at two data sizes.

The cell side is meshsearch's one tuning parameter. A fine grid holds few
objects per cell and many cells; a coarse one the reverse, and the offset mask
holds 8 entries per cell, so the cell count and the memory with it fall as the
cube of the factor. The radius is held at twice the mean separation so that the
answer stays the same size, about 33 neighbours, at every cell side.

The tables this produces are in benchmark.md, which reads them.

    python docs/benchmark_cellsize.py --sweep          # every size and cell side
    python docs/benchmark_cellsize.py --one-sweep 1000000 3   # one, as JSON
"""

import argparse
import gc
import json
import subprocess
import sys
import time

import numpy as np

N_QUERIES = 1000
K = 10
SEED = 20260921
FACTORS = [1, 2, 3, 4]
SIZES = [1_000_000, 10_000_000]


def points(n):
    rng = np.random.default_rng(SEED)
    return rng.random(n), rng.random(n), rng.random(n)


def query_points():
    rng = np.random.default_rng(SEED + 1)
    return rng.random((N_QUERIES, 3))


def rss_mib():
    import psutil
    return psutil.Process().memory_info().rss / 2 ** 20


def timed(fn, repeats=3):
    return float(np.median([_once(fn) for _ in range(repeats)]))


def _once(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def measure(grid, queries, radius):
    return {
        "nn_us": timed(lambda: [grid.nearest_object(*p) for p in queries]) * 1e3,
        "knn_us": timed(lambda: [grid.nearest_objects(K, *p) for p in queries]) * 1e3,
        "radius_us": timed(lambda: [grid.close_objects(*p, radius) for p in queries]) * 1e3,
        "radius_count": float(np.mean([grid.close_objects(*p, radius).size for p in queries])),
    }


def run_sweep(n, factor):
    import meshsearch
    x, y, z = points(n)
    queries = query_points()
    mps = n ** (-1 / 3)
    cellsize = factor * mps
    radius = 2 * mps                      # the answer size is held constant

    gc.collect()
    before = rss_mib()
    t = time.perf_counter()
    grid = meshsearch.MeshGrid(x, y, z, cellsize)
    build = time.perf_counter() - t
    memory = rss_mib() - before

    cells = grid.n_cells[0] * grid.n_cells[1] * grid.n_cells[2]
    result = {"n": n, "factor": factor, "cellsize": cellsize, "build_s": build,
              "memory_mib": memory, "n_cells": list(grid.n_cells), "cells": cells,
              "objects_per_cell": n / cells}
    result.update(measure(grid, queries, radius))
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
    parser.add_argument("--one-sweep", nargs=2, type=int, metavar=("N", "FACTOR"))
    args = parser.parse_args()

    if args.one_sweep:
        print(json.dumps(run_sweep(*args.one_sweep)))
    elif args.sweep:
        rows = []
        for n in SIZES:
            for factor in FACTORS:
                sys.stderr.write(f"  sweep n={n:,} cellsize={factor} mps ... ")
                sys.stderr.flush()
                row = spawn(["--one-sweep", str(n), str(factor)])
                sys.stderr.write(f"{row['memory_mib']:,.0f} MiB\n")
                rows.append(row)
        print(json.dumps(rows, indent=2))
    else:
        parser.error("choose --sweep or --one-sweep N FACTOR")


if __name__ == "__main__":
    main()
