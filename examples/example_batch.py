"""The batched queries, which exist only in the Python bindings.

Crossing the Python boundary costs a few microseconds per call, which is real
work when the query itself takes one or two. The batched forms hand the whole
array of query points across once. A C++ caller writes a loop and pays nothing,
which is why there is no batched API in the C++ library.

    pip install .
    python examples/example_batch.py
"""

import numpy as np

import meshsearch

# The grid, and a set of query points to ask about.
rng = np.random.default_rng(7)
n = 200_000
x, y, z = rng.random(n), rng.random(n), rng.random(n)
grid = meshsearch.MeshGrid(x, y, z, 4 * n ** (-1 / 3), [[0.0, 1.0]] * 3)

# Query points go in as three arrays, the same shape as the constructor takes.
# They must be numpy arrays rather than lists: an array argument is what selects
# the batched overload, and a list would be ambiguous with a single point.
queries = rng.random((50_000, 3))
qx = np.ascontiguousarray(queries[:, 0])
qy = np.ascontiguousarray(queries[:, 1])
qz = np.ascontiguousarray(queries[:, 2])

print(f"{len(grid)} objects, {len(qx)} query points\n")

# 1. The nearest object to each query point: one index per query, in query
#    order. The result indexes straight back into the coordinate arrays.
nearest = grid.nearest_object(qx, qy, qz)
print(f"nearest_object -> {nearest.dtype} {nearest.shape}")
separation = np.sqrt((x[nearest] - qx) ** 2 + (y[nearest] - qy) ** 2
                     + (z[nearest] - qz) ** 2)
print(f"  mean distance to the nearest object: {separation.mean():.5f}")

# 2. The five nearest to each query point, nearest first. This one is
#    rectangular -- every query returns exactly n -- so it comes back as a
#    single (n_query, n) array and needs no offsets.
five = grid.nearest_objects(5, qx, qy, qz)
print(f"nearest_objects -> {five.dtype} {five.shape}")
print(f"  the fifth-nearest is further than the first for every query: "
      f"{bool(np.all(np.sqrt((x[five[:, 4]] - qx) ** 2 + (y[five[:, 4]] - qy) ** 2 + (z[five[:, 4]] - qz) ** 2) >= separation))}")

# 3. A shell around each query point. The count varies per query, so the result
#    cannot be rectangular. It comes back in compressed-row form: one flat
#    array of every result, and an offsets array saying where each query's
#    results begin and end.
radius = 0.02
indices, offsets = grid.close_objects(qx, qy, qz, radius)
print(f"close_objects -> indices {indices.dtype} {indices.shape}, "
      f"offsets {offsets.dtype} {offsets.shape}")

# The results of query i are one slice, with no copying:
i = 12
print(f"  query {i} has {offsets[i + 1] - offsets[i]} neighbours: "
      f"{indices[offsets[i]:offsets[i + 1]]}")

# The per-query count is a difference of offsets -- no loop, no unpacking.
counts = np.diff(offsets)
print(f"  neighbours per query: min {counts.min()}, mean {counts.mean():.2f}, "
      f"max {counts.max()}, total {len(indices)}")
print(f"  queries with no neighbour at all: {int(np.sum(counts == 0))}")

# What the flat form makes easy. Pairing each result with the query it belongs
# to turns the whole batch into two flat arrays of equal length, after which
# every per-query quantity is one vectorised expression over all results at
# once. With a list of arrays this would be a Python loop over the queries --
# and one Python object per query, which is the cost the batch just removed.
query_of = np.repeat(np.arange(len(qx)), counts)

dx = x[indices] - qx[query_of]
dy = y[indices] - qy[query_of]
dz = z[indices] - qz[query_of]
distance = np.sqrt(dx * dx + dy * dy + dz * dz)

# The mean distance to the neighbours of each query. bincount sums over the
# results belonging to each query; the guarded divide leaves a query with no
# neighbours at zero rather than dividing by it.
summed = np.bincount(query_of, weights=distance, minlength=len(qx))
mean_distance = np.divide(summed, counts, out=np.zeros(len(qx)), where=counts > 0)

print(f"\n  mean neighbour distance, averaged over queries that have one: "
      f"{mean_distance[counts > 0].mean():.5f}")
print(f"  and the shell holds them all: "
      f"{bool(np.all(distance <= radius + 1e-12))}")

# The same trick answers questions that span queries. Which object turns up as
# a neighbour most often across the whole batch?
busiest = np.bincount(indices, minlength=n).argmax()
print(f"  object {busiest} is a neighbour of "
      f"{int(np.sum(indices == busiest))} different queries")

# Errors name the query that failed, and nothing partial is returned.
bad_x = qx.copy()
bad_x[99] = 5.0
try:
    grid.nearest_object(bad_x, qy, qz)
except meshsearch.Error as e:
    print(f"\na query point outside the box raised Error: {e}")

# The index-taking overloads have no batched form: their input is a single
# index, so there is no per-call array to amortise anything over.
print(f"five nearest to object {nearest[0]}: {grid.nearest_objects(5, int(nearest[0]))}")
