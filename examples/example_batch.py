"""Batched queries: many query points answered in one call.

Crossing the Python boundary costs a few microseconds per call, which is real
work when the query itself takes one or two. The batched forms hand the whole
array of query points across once, and a large batch is answered in an order
that follows the grid's cells, which is faster again. Results always come back
in query order.

    pip install .
    python examples/example_batch.py
"""

import numpy as np

import meshsearch

# The grid, and a set of query points to ask about.
rng = np.random.default_rng(7)
n = 200_000
x, y, z = rng.random(n), rng.random(n), rng.random(n)
grid = meshsearch.MeshGrid(x, y, z, 4 * n ** (-1 / 3), limits=[[0.0, 1.0]] * 3)

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

# 3. Only the number of objects within a radius of each query point. No index
#    is collected, so this is cheaper than close_objects when the counts are
#    all that is needed. Counts are int64.
radius = 0.02
within = grid.count_objects(qx, qy, qz, radius)
print(f"count_objects -> {within.dtype} {within.shape}, mean {within.mean():.2f}")

# 4. A shell around each query point. The count varies per query, so the result
#    cannot be rectangular. It comes back in compressed-row form: one flat
#    array of every result, and an offsets array saying where each query's
#    results begin and end.
indices, offsets = grid.close_objects(qx, qy, qz, radius)
print(f"close_objects -> indices {indices.dtype} {indices.shape}, "
      f"offsets {offsets.dtype} {offsets.shape}")

# The results of query i are one slice, with no copying:
i = 12
print(f"  query {i} has {offsets[i + 1] - offsets[i]} neighbours: "
      f"{indices[offsets[i]:offsets[i + 1]]}")

# The per-query count is a difference of offsets -- no loop, no unpacking --
# and it is what count_objects returned.
counts = np.diff(offsets)
assert np.array_equal(counts, within)
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

# Repeated batches of the same shape can write into one array instead of
# allocating a new one each time: pass it as out=, and the call fills it and
# returns it. The array must already be exactly the result -- uint32 of shape
# (n_query, n) for nearest_objects, int64 of shape (n_query,) for
# count_objects, C-contiguous -- or the call raises; it is never converted.
# close_objects has no out=, because its length is only known afterwards.
neighbours = np.empty((len(qx), 5), dtype=np.uint32)
tally = np.empty(len(qx), dtype=np.int64)
for r in (0.01, 0.02, 0.04):
    grid.count_objects(qx, qy, qz, r, out=tally)
    print(f"  within {r}: {int(tally.sum())} neighbours over all queries")
grid.nearest_objects(5, qx, qy, qz, out=neighbours)
assert np.array_equal(neighbours, five)

# The GIL is released while a batch runs, so other threads can work, including
# other queries on the same grid. None may modify the grid (add_object,
# remove_object) until the batch has returned.

# Errors name the query that failed, and nothing partial is returned.
bad_x = qx.copy()
bad_x[99] = 5.0
try:
    grid.nearest_object(bad_x, qy, qz)
except meshsearch.Error as e:
    print(f"\na query point outside the box raised Error: {e}")

# Every query also has a batch over objects already in the grid: an array of
# indices in place of the three coordinate arrays. Each object is excluded from
# its own answer. Any integer dtype is accepted, such as the uint32 the queries
# return or numpy's default int64, and an index that is negative, too large or
# removed raises IndexError naming the query, rather than being wrapped.
objects = nearest[:1000]
partner = grid.nearest_object(objects)
ring, ring_offsets = grid.close_objects(objects, radius)
around = grid.count_objects(objects, radius)
print(f"\nobject {objects[0]}: nearest other object {partner[0]}, "
      f"{around[0]} objects within {radius}")
assert np.array_equal(np.diff(ring_offsets), around)
