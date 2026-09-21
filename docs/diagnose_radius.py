"""Why closeObjects stays slower than cKDTree once layout is accounted for.

Diagnosis only: nothing here changes or tunes the library. Four measurements,
described in the radius section of benchmark.md.

  A  the layer range: how many cells a shell query visits, against how many
     can hold a member, derived from the algorithm in src/MeshGrid.cpp
  B  whether the cost is the fixed scan or the results, by varying the radius
     with the scan held constant and vice versa
  C  the result vector, by driving the result count up
  D  how many distance evaluations cKDTree needs for the same query

    python docs/diagnose_radius.py
"""

import math
import time

import numpy as np

SEED = 20260921
K_MASK = 12          # layers of offsets to enumerate; far past anything used here


# ---------------------------------------------------------------- the mask

def mask_layers(reach=16):
    """Offsets grouped by layer, exactly as the constructor groups them:
    per-axis gap d = max(|i|-1, 0), layer = floor(sqrt(dx^2+dy^2+dz^2))."""
    layers = {}
    for i in range(-reach, reach + 1):
        dx = max(abs(i) - 1, 0)
        for j in range(-reach, reach + 1):
            dy = max(abs(j) - 1, 0)
            for k in range(-reach, reach + 1):
                dz = max(abs(k) - 1, 0)
                layer = int(math.floor(math.sqrt(dx * dx + dy * dy + dz * dz)))
                layers.setdefault(layer, []).append((i, j, k))
    return layers


def layers_scanned(rmax_c, rmin_c, n_layers):
    """The layer indices closeObjects actually visits, for a radius expressed
    in cell sides. Mirrors the bounds and the in-loop break in the source."""
    lower = math.floor(rmin_c) - 4
    min_idx = max(int(lower), 0)
    upper = math.ceil(rmax_c) + 3
    max_idx = min(int(upper), n_layers)

    visited = []
    for m in range(min_idx, max_idx):
        if m > 4 and ((m - 3) ** 2 > rmax_c ** 2):
            break
        visited.append(m)
    return visited


def cell_distance_range(offset, frac):
    """Min and max distance, in cell sides, from a query point sitting at
    `frac` inside its own cell to any point of the cell at `offset`."""
    lo = hi = 0.0
    for o, f in zip(offset, frac):
        near = max(o - f, 0.0, f - o - 1.0)          # gap along this axis
        far = max(abs(o - f), abs(o + 1.0 - f))
        lo += near * near
        hi += far * far
    return math.sqrt(lo), math.sqrt(hi)


def part_a(rmax_c, rmin_c=0.0, samples=200):
    layers = mask_layers()
    n_layers = max(layers) + 1
    visited = layers_scanned(rmax_c, rmin_c, n_layers)
    cells = [o for m in visited for o in layers[m]]

    rng = np.random.default_rng(SEED)
    useful = wasted = 0
    for frac in rng.random((samples, 3)):
        for offset in cells:
            lo, hi = cell_distance_range(offset, frac)
            if lo <= rmax_c and hi >= rmin_c:
                useful += 1
            else:
                wasted += 1
    return {
        "rmax_over_cellsize": rmax_c,
        "layers_scanned": visited,
        "cells_visited": len(cells),
        "cells_that_can_hold_a_member": useful / samples,
        "cells_that_cannot": wasted / samples,
        "necessary_layers": [m for m in visited if m <= rmax_c],
    }


# ------------------------------------------------------------- empirical

def build(n, factor, order="cell"):
    import meshsearch
    rng = np.random.default_rng(SEED)
    x, y, z = rng.random(n), rng.random(n), rng.random(n)
    mps = n ** (-1 / 3)
    cellsize = factor * mps
    if order == "cell":
        probe = meshsearch.MeshGrid(x, y, z, cellsize)
        lims, nc = probe.lims, probe.n_cells
        idx = [np.clip(np.floor((c - lo) / cellsize).astype(np.int64), 0, n_ - 1)
               for c, (lo, _), n_ in zip((x, y, z), lims, nc)]
        order_ = np.argsort(idx[0] * nc[1] * nc[2] + idx[1] * nc[2] + idx[2], kind="stable")
        x, y, z = (np.ascontiguousarray(a[order_]) for a in (x, y, z))
        del probe
    return meshsearch.MeshGrid(x, y, z, cellsize), mps, cellsize


def timed(fn, repeats=3):
    return float(np.median([_once(fn) for _ in range(repeats)]))


def _once(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def queries(n=1000):
    return np.random.default_rng(SEED + 1).random((n, 3))


# ------------------------------------------------------------ cKDTree work

def ckdtree_distance_evaluations(data, points, r):
    """Upper bound on the points cKDTree tests for a ball query.

    Descends the exposed tree, narrowing a box at each split. scipy's own node
    boxes are tighter than the ones a split rebuilds, so it prunes at least as
    hard as this does and the count is an upper bound.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(data)
    root_lo, root_hi = data.min(axis=0), data.max(axis=0)

    def descend(node, lo, hi, p):
        gap = np.maximum(np.maximum(lo - p, p - hi), 0.0)
        if np.dot(gap, gap) > r * r:
            return 0
        if node.split_dim == -1:
            return len(node.indices)
        d, s = node.split_dim, node.split
        hi_lesser = hi.copy()
        hi_lesser[d] = s            # the lesser child is bounded above by the split
        lo_greater = lo.copy()
        lo_greater[d] = s           # the greater child is bounded below by it
        return (descend(node.lesser, lo, hi_lesser, p)
                + descend(node.greater, lo_greater, hi, p))

    root = tree.tree
    return [descend(root, root_lo.copy(), root_hi.copy(), p) for p in points], tree


# ----------------------------------------------------------------- report

def main():
    n, factor = 10_000_000, 4
    mps = n ** (-1 / 3)
    rmax = 2 * mps
    cellsize = factor * mps
    rmax_c = rmax / cellsize

    print("=" * 72)
    print(f"A. the layer range   (n={n:,}, cellsize={factor} mps, "
          f"Rmax={rmax:.6f} = {rmax_c:g} cellsize)")
    print("=" * 72)
    a = part_a(rmax_c)
    print(f"  layers scanned              {a['layers_scanned']}")
    print(f"  layers that can hold a hit  {a['necessary_layers']}")
    print(f"  cells visited               {a['cells_visited']:,}")
    print(f"  of those, can hold a member {a['cells_that_can_hold_a_member']:.1f} (mean over query positions)")
    print(f"  cannot hold a member        {a['cells_that_cannot']:.1f}")
    print(f"  wasted fraction of cells    {a['cells_that_cannot']/a['cells_visited']:.1%}")

    print()
    print("  the same, over a range of Rmax/cellsize:")
    print("    Rmax/c  layers          cells  can hold  wasted")
    for rc in (0.25, 0.5, 1.0, 2.0, 4.0):
        r = part_a(rc, samples=60)
        print(f"    {rc:6.2f}  {str(r['layers_scanned']):14s} {r['cells_visited']:6,}"
              f"  {r['cells_that_can_hold_a_member']:8.1f}  {r['cells_that_cannot']/r['cells_visited']:6.1%}")

    grid, mps, cellsize = build(n, factor)
    qs = queries()

    print()
    print("=" * 72)
    print("B. is the cost the fixed scan, or the results?")
    print("=" * 72)
    print("   Rmax is varied while the layer range is held at the same 4 layers")
    print("   (max_mask_idx = ceil(Rmax/cellsize) + 3 = 4 for every Rmax <= 1 cellsize)")
    print()
    print("    Rmax         Rmax/c  results  time_us")
    for mult in (1e-9, 0.25, 0.5, 1.0, 1.5, 2.0):
        r = mult * mps
        t = timed(lambda: [grid.close_objects(*p, r) for p in qs]) * 1e3
        cnt = np.mean([grid.close_objects(*p, r).size for p in qs])
        print(f"    {mult:4.2f} mps    {r/cellsize:5.3f}  {cnt:7.1f}  {t:7.2f}")

    print()
    print("   for scale, the same grid:")
    t_nn = timed(lambda: [grid.nearest_object(*p) for p in qs]) * 1e3
    print(f"    nearest_object (scans layer 0 only)          {t_nn:7.2f} us")
    t_cell = timed(lambda: [grid.get_objects_in_cell(*p) for p in qs]) * 1e3
    print(f"    get_objects_in_cell (1 cell, ~64 objects)    {t_cell:7.2f} us")

    print()
    print("=" * 72)
    print("C. the result vector")
    print("=" * 72)
    print("    Rmax/c  results   time_us   us per result")
    for mult in (2.0, 4.0, 8.0, 16.0):
        r = mult * mps
        t = timed(lambda: [grid.close_objects(*p, r) for p in qs]) * 1e3
        cnt = np.mean([grid.close_objects(*p, r).size for p in qs])
        a = part_a(r / cellsize, samples=40)
        print(f"    {r/cellsize:6.2f}  {cnt:7.0f}   {t:7.2f}   {t/cnt:9.4f}"
              f"   (cells visited {a['cells_visited']:,})")

    print()
    print("=" * 72)
    print("D. how much work does cKDTree do for the same query?")
    print("=" * 72)
    nd = 1_000_000
    rng = np.random.default_rng(SEED)
    data = np.ascontiguousarray(np.stack([rng.random(nd), rng.random(nd), rng.random(nd)], axis=1))
    mps_d = nd ** (-1 / 3)
    r = 2 * mps_d
    sample = queries(50)
    counts, tree = ckdtree_distance_evaluations(data, sample, r)
    hits = np.mean([len(tree.query_ball_point(p, r)) for p in sample])
    print(f"  n = {nd:,}, Rmax = 2 mps, {len(sample)} query points")
    print(f"    cKDTree points tested (upper bound)   {np.mean(counts):8.0f}")
    print(f"    hits                                  {hits:8.1f}")
    print(f"    tested per hit                        {np.mean(counts)/hits:8.1f}")
    a6 = part_a(r / (4 * mps_d), samples=60)
    per_cell = nd / ( (1/(4*mps_d))**3 )
    print(f"    meshsearch at 4 mps: cells visited    {a6['cells_visited']:8,}")
    print(f"    objects per cell                      {per_cell:8.0f}")
    print(f"    meshsearch distance tests             {a6['cells_visited']*per_cell:8.0f}")
    print(f"    tested per hit                        {a6['cells_visited']*per_cell/hits:8.1f}")


def part_e():
    """Fit cost = cells_visited * (per_cell + objects_per_cell * per_test) at
    1e7 over the four cell sides, and check it against the measurements."""
    n = 10_000_000
    mps = n ** (-1 / 3)
    rmax = 2 * mps
    qs = queries()
    rows = []
    for factor in (1, 2, 3, 4):
        grid, _, cellsize = build(n, factor)
        cells = part_a(rmax / cellsize, samples=40)["cells_visited"]
        per_cell_objects = n / (grid.n_cells[0] * grid.n_cells[1] * grid.n_cells[2])
        t = timed(lambda: [grid.close_objects(*p, rmax) for p in qs]) * 1e3
        rows.append((factor, cells, per_cell_objects, t))
        del grid
    A = np.array([[c, c * o] for _, c, o, _ in rows])
    y = np.array([t * 1e3 for *_, t in rows])          # nanoseconds
    (per_cell, per_test), *_ = np.linalg.lstsq(A, y, rcond=None)
    print(f"  fitted: {per_cell:.1f} ns per cell visited, {per_test:.2f} ns per distance test")
    print("    cellsize   cells  obj/cell  measured_us  predicted_us  error")
    for (factor, c, o, t), pred in zip(rows, (A @ [per_cell, per_test]) / 1e3):
        print(f"    {factor} mps  {c:7,}  {o:8.1f}  {t:11.2f}  {pred:12.2f}  {(pred-t)/t:+5.1%}")
    return per_cell, per_test


if __name__ == "__main__":
    main()
    print()
    print("=" * 72)
    print("E. a cost model, fitted and checked")
    print("=" * 72)
    part_e()
