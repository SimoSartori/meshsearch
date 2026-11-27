"""Bit-for-bit reproducibility of every query across platforms.

Each case below hashes the bytes of one result array, in the exact order the
library returns it, and the hash must equal the one recorded in
tests/determinism_hashes.json: the same inputs give the same results, in the
same order, on every platform the wheels are built for. The header leaves the
order of objects within a cell unspecified, so this file pins the current
order, which is the one otswap and bitvf build their own hashes on. A change
that reorders results on purpose (a speed refactor) re-records the file, and
re-recording is then a visible decision in the diff.

The inputs are identical in bytes everywhere. They come from a fixed 64-bit
linear congruential generator in pure Python, mapped to doubles by exact
division by 2^53 and then by single IEEE operations: no libm call and no NumPy
random generator. Cell sides, limits and radii are exact binary fractions, so
the geometry does not depend on a rounded division either.

The batch forms take workers=, the number of threads they may use, and the
results must not depend on it: the recorded hashes are computed with one
worker and checked again with several. These batches are small, below the size
at which a batch is split over threads, so that second check holds the keyword
to the same results; a larger batch, hashed but not recorded, is compared
across worker counts to exercise the threads themselves.

Run as a script, this file prints the reference file's content as JSON.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import meshsearch

REFERENCE = Path(__file__).resolve().parent / "determinism_hashes.json"

N_QUERY = 40
N_INDEX = 25
N_NEAREST = 7


class Lcg:
    """Knuth's MMIX generator, x <- 6364136223846793005 x + 1442695040888963407
    mod 2^64; a double from the top 53 bits."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next(self):
        self.state = (6364136223846793005 * self.state + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.state

    def unit(self):
        return (self.next() >> 11) / 9007199254740992.0

    def uniform(self, lo, hi):
        return lo + self.unit() * (hi - lo)


def uniform_points(n, g):
    return np.array([[g.unit() for _ in range(3)] for _ in range(n)])


def clustered_points(n, g):
    centres = [(0.25, 0.25, 0.25), (0.75, 0.25, 0.5), (0.5, 0.75, 0.75), (0.25, 0.75, 0.25)]
    pts = []
    for i in range(n):
        c = centres[i % len(centres)]
        pts.append([c[a] + g.uniform(-0.0625, 0.0625) for a in range(3)])
    return np.array(pts)


# Dense: many objects per cell. Sparse: most cells empty, so queries walk many
# of them. Clustered: dense knots in an otherwise empty box. The sizes are
# small, so that the whole file runs in a few seconds.
REGIMES = {
    "dense": dict(points=uniform_points, n=500, cellsize=0.25, rmax=0.25, rmin=0.125),
    "sparse": dict(points=uniform_points, n=120, cellsize=0.0625, rmax=0.25, rmin=0.125),
    "clustered": dict(points=clustered_points, n=400, cellsize=0.125, rmax=0.1875, rmin=0.0625),
}

# A periodic axis needs limits and a period that is a whole number of cells,
# which a unit box and a cell side 2^-k satisfy exactly. Dense and
# non-periodic is the one case with no limits, to hash the fitted geometry.
BOUNDARIES = {
    "open": (False, False, False),
    "periodic": (True, True, True),
    "mixed": (True, False, True),
}


def digest(a):
    a = np.asarray(a)
    if a.dtype == np.bool_:
        a = a.astype("|u1")
    elif a.dtype.kind == "f":
        a = a.astype("<f8")
    elif a.dtype.kind in "iu":
        a = a.astype("<i8")
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def case(boundary, regime, seed, workers=1):
    spec = REGIMES[regime]
    periodic = BOUNDARIES[boundary]
    g = Lcg(seed)
    p = spec["points"](spec["n"], g)
    x, y, z = (np.ascontiguousarray(p[:, a]) for a in range(3))
    fitted = boundary == "open" and regime == "dense"
    kw = {} if fitted else {"limits": [[0.0, 1.0]] * 3}
    grid = meshsearch.MeshGrid(x, y, z, spec["cellsize"], periodic=periodic, **kw)

    q = np.array([[g.uniform(0.125, 0.875) for _ in range(3)] for _ in range(N_QUERY)])
    qx, qy, qz = (np.ascontiguousarray(q[:, a]) for a in range(3))
    wide = np.array([[g.uniform(-0.25, 1.25) for _ in range(3)] for _ in range(N_QUERY)])
    # Includes indices that were never issued: is_alive answers False for them.
    idx = np.array([g.next() % (spec["n"] + 4) for _ in range(N_INDEX)], dtype=np.int64)
    live = idx[idx < spec["n"]].astype(np.uint32)
    rmax, rmin = spec["rmax"], spec["rmin"]

    out = {}

    def put(field, a):
        out[f"{boundary}.{regime}.{field}"] = digest(a)

    def stacked(arrays):
        return np.concatenate([np.asarray(a, dtype=np.int64).ravel() for a in arrays])

    put("lims", grid.lims)
    put("n_cells", grid.n_cells)
    put("cellsize", [grid.cellsize])

    pts = list(zip(qx, qy, qz))
    put("nearest_object.point.single", [grid.nearest_object(*c) for c in pts])
    put("nearest_object.point.batch", grid.nearest_object(qx, qy, qz, workers=workers))
    put("nearest_objects.point.single", stacked([grid.nearest_objects(N_NEAREST, *c) for c in pts]))
    put("nearest_objects.point.batch", grid.nearest_objects(N_NEAREST, qx, qy, qz, workers=workers))
    for tag, args in (("rmax", (rmax,)), ("rmin", (rmax, rmin))):
        singles = [grid.close_objects(*c, *args) for c in pts]
        put(f"close_objects.point.{tag}.single", stacked(singles))
        put(f"close_objects.point.{tag}.single_sizes", [len(s) for s in singles])
        batch = grid.close_objects(qx, qy, qz, *args, workers=workers)
        put(f"close_objects.point.{tag}.batch_indices", batch.indices)
        put(f"close_objects.point.{tag}.batch_offsets", batch.offsets)
        put(f"count_objects.point.{tag}.single", [grid.count_objects(*c, *args) for c in pts])
        put(f"count_objects.point.{tag}.batch", grid.count_objects(qx, qy, qz, *args, workers=workers))
    put("contains.single", [grid.contains(*c) for c in wide])
    put("contains.batch", grid.contains(wide[:, 0].copy(), wide[:, 1].copy(), wide[:, 2].copy()))

    put("is_alive.single", [grid.is_alive(int(i)) for i in idx])
    put("is_alive.batch", grid.is_alive(idx))
    put("nearest_object.index.single", [grid.nearest_object(int(i)) for i in live])
    put("nearest_object.index.batch", grid.nearest_object(live, workers=workers))
    put("nearest_objects.index.single", stacked([grid.nearest_objects(N_NEAREST, int(i)) for i in live]))
    put("nearest_objects.index.batch", grid.nearest_objects(N_NEAREST, live, workers=workers))
    for tag, args in (("rmax", (rmax,)), ("rmin", (rmax, rmin))):
        singles = [grid.close_objects(int(i), *args) for i in live]
        put(f"close_objects.index.{tag}.single", stacked(singles))
        put(f"close_objects.index.{tag}.single_sizes", [len(s) for s in singles])
        batch = grid.close_objects(live, *args, workers=workers)
        put(f"close_objects.index.{tag}.batch_indices", batch.indices)
        put(f"close_objects.index.{tag}.batch_offsets", batch.offsets)
        put(f"count_objects.index.{tag}.single", [grid.count_objects(int(i), *args) for i in live])
        put(f"count_objects.index.{tag}.batch", grid.count_objects(live, *args, workers=workers))
    return out


def compute(workers=1):
    out = {}
    seed = 1
    for boundary in BOUNDARIES:
        for regime in REGIMES:
            out.update(case(boundary, regime, seed, workers))
            seed += 1
    return out


def test_hashes_match_the_reference():
    expected = json.loads(REFERENCE.read_text())["hashes"]
    got = compute()
    differ = sorted(k for k in expected if k in got and got[k] != expected[k])
    assert not differ, f"these outputs changed: {differ}"
    assert set(got) == set(expected), (
        f"the cases differ from the recorded ones: new {sorted(set(got) - set(expected))}, "
        f"gone {sorted(set(expected) - set(got))}")


@pytest.mark.parametrize("workers", [2, 7, -1])
def test_hashes_match_the_reference_with_several_workers(workers):
    expected = json.loads(REFERENCE.read_text())["hashes"]
    got = compute(workers)
    differ = sorted(k for k in expected if k in got and got[k] != expected[k])
    assert not differ, f"these outputs changed with workers={workers}: {differ}"
    assert set(got) == set(expected)


def test_a_batch_split_over_threads_hashes_as_with_one_worker():
    g = Lcg(99)
    p = clustered_points(3000, g)
    x, y, z = (np.ascontiguousarray(p[:, a]) for a in range(3))
    q = np.array([[g.uniform(0.125, 0.875) for _ in range(3)] for _ in range(4000)])
    qx, qy, qz = (np.ascontiguousarray(q[:, a]) for a in range(3))
    live = np.array([g.next() % 3000 for _ in range(4000)], dtype=np.uint32)
    for periodic in (False, True):
        grid = meshsearch.MeshGrid(x, y, z, 0.0625, limits=[[0.0, 1.0]] * 3, periodic=periodic)

        def hashes(workers):
            close = grid.close_objects(qx, qy, qz, 0.09375, 0.03125, workers=workers)
            closei = grid.close_objects(live, 0.09375, workers=workers)
            return [digest(a) for a in (grid.nearest_object(qx, qy, qz, workers=workers),
                                        grid.nearest_objects(N_NEAREST, live, workers=workers),
                                        close.indices, close.offsets, closei.indices, closei.offsets,
                                        grid.count_objects(live, 0.09375, 0.03125, workers=workers))]

        one = hashes(1)
        for workers in (2, 7, -1):
            assert hashes(workers) == one, f"periodic={periodic}, workers={workers}"


def test_the_single_and_batch_forms_hash_alike():
    got = compute()
    pairs = [(k, k.replace(".single", ".batch")) for k in got if k.endswith(".single")]
    pairs += [(k, k.replace(".single", ".batch_indices")) for k in got
              if k.startswith("close_objects") and k.endswith(".single")]
    pairs = [(a, b) for a, b in pairs if b in got]
    assert len(pairs) > 20
    for a, b in pairs:
        assert got[a] == got[b], f"{a} differs from {b}"


if __name__ == "__main__":
    print(json.dumps({"hashes": compute()}, indent=1, sort_keys=True))
