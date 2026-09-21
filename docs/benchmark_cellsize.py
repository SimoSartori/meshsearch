"""Two follow-up measurements on meshsearch, both described in benchmark.md.

1. The cell side sweep. The first benchmark used one mean separation, which
   gives about one object per cell. Grids in use are built coarser -- OT builds
   at four mean separations, about 64 objects per cell -- and the mask holds
   8 entries per cell, so the cell count and the memory with it fall as the
   cube of the factor.

2. The locality test. Build the same points twice at the same cell side, once
   in input order and once sorted by the cell each point falls into. Same n,
   same cell count, same number of distance tests; only the order of the
   per-cell allocations and of the coordinate accesses differs.

    python docs/benchmark_cellsize.py --sweep
    python docs/benchmark_cellsize.py --locality
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


def cell_of(grid, x, y, z):
    """The linear cell index of every point, in numpy.

    Replicates what get_cell does, so that a whole array can be binned at once;
    checked against get_cell on a sample before use.
    """
    lims, n_cells = grid.lims, grid.n_cells
    idx = []
    for coord, (lo, _), n in zip((x, y, z), lims, n_cells):
        i = np.floor((coord - lo) / grid.cellsize).astype(np.int64)
        idx.append(np.clip(i, 0, n - 1))
    return (idx[0] * n_cells[1] * n_cells[2] + idx[1] * n_cells[2] + idx[2])


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


def run_locality(n, factor):
    import meshsearch
    x, y, z = points(n)
    queries = query_points()
    mps = n ** (-1 / 3)
    cellsize = factor * mps
    radius = 2 * mps

    # the permutation that puts the points in cell order
    probe = meshsearch.MeshGrid(x, y, z, cellsize)
    cells = cell_of(probe, x, y, z)
    sample = np.random.default_rng(7).integers(0, n, 200)
    agree = all(int(cells[i]) == probe.get_cell(x[i], y[i], z[i]) for i in sample)
    order = np.argsort(cells, kind="stable")
    del probe
    gc.collect()

    out = {"n": n, "factor": factor, "cellsize": cellsize,
           "binning_matches_get_cell": bool(agree)}

    for label, (px, py, pz) in (("input_order", (x, y, z)),
                                ("cell_order", (x[order], y[order], z[order]))):
        px, py, pz = (np.ascontiguousarray(a) for a in (px, py, pz))
        gc.collect()
        before = rss_mib()
        t = time.perf_counter()
        grid = meshsearch.MeshGrid(px, py, pz, cellsize)
        build = time.perf_counter() - t
        row = {"build_s": build, "memory_mib": rss_mib() - before}
        row.update(measure(grid, queries, radius))

        # the two grids must agree geometrically, indices aside
        d = []
        for p in queries[:100]:
            i = grid.nearest_object(*p)
            d.append((px[i] - p[0]) ** 2 + (py[i] - p[1]) ** 2 + (pz[i] - p[2]) ** 2)
        row["nearest_dist_checksum"] = float(np.sum(np.sqrt(d)))
        out[label] = row
        del grid
        gc.collect()

    return out


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
    parser.add_argument("--locality", action="store_true")
    parser.add_argument("--one-sweep", nargs=2, type=int)
    parser.add_argument("--one-locality", nargs=2, type=int)
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    if args.one_sweep:
        print(json.dumps(run_sweep(*args.one_sweep)))
    elif args.one_locality:
        print(json.dumps(run_locality(*args.one_locality)))
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
    elif args.locality:
        rows = []
        for n in SIZES:
            sys.stderr.write(f"  locality n={n:,} cellsize={args.factor} mps\n")
            rows.append(spawn(["--one-locality", str(n), str(args.factor)]))
        print(json.dumps(rows, indent=2))
    else:
        parser.error("choose --sweep or --locality")


if __name__ == "__main__":
    main()
