"""A short tour of meshsearch from Python: build a grid over a set of points,
ask the three kinds of neighbour question, modify the grid, and handle a bad
query.

    pip install .
    python examples/example.py
"""

import numpy as np

import meshsearch

# The points, as three arrays of the same length. An object is identified by
# its position in them, and the arrays are copied into the grid.
rng = np.random.default_rng(42)
n = 5000
x, y, z = rng.random(n), rng.random(n), rng.random(n)

# The cell side sets the cost of every query: too large and each cell holds
# many objects to test, too small and a query walks many cells. The mean
# separation is a reasonable starting point.
#
# Coordinates, cellsize and radii must share one unit. The library does not
# impose one, so whatever the points are in, the radii are in too.
cellsize = n ** (-1 / 3)

# With no limits given, the box is fitted to the data and then grown
# symmetrically to a whole number of cells, so grid.lims can be slightly wider
# than the points. Pass limits explicitly to control it:
#     meshsearch.MeshGrid(x, y, z, cellsize, [[0, 1], [0, 1], [0, 1]])
grid = meshsearch.MeshGrid(x, y, z, cellsize)

print(f"{len(grid)} objects, {'x'.join(str(c) for c in grid.n_cells)} cells "
      f"of {grid.cellsize:.4f}")
for axis, (lo, hi) in zip("xyz", grid.lims):
    print(f"  {axis} [{lo:.4f}, {hi:.4f}]")
print()

# 1. The object nearest a point.
nearest = grid.nearest_object(0.5, 0.5, 0.5)
print(f"nearest to the box centre: object {nearest} at "
      f"({x[nearest]:.4f}, {y[nearest]:.4f}, {z[nearest]:.4f})")

# 2. The n nearest, nearest first. Index results are uint32 arrays, so they
#    index straight back into the coordinate arrays.
ten = grid.nearest_objects(10, 0.5, 0.5, 0.5)
print(f"ten nearest to the centre: {ten}")
print("their distances:",
      np.round(np.sqrt((x[ten] - 0.5) ** 2 + (y[ten] - 0.5) ** 2
                       + (z[ten] - 0.5) ** 2), 4))

# 3. Every object in a spherical shell, rmin <= d <= rmax, in no particular
#    order. rmin defaults to 0, giving a ball. A shell reaching outside the box
#    is truncated rather than rejected.
shell = grid.close_objects(0.5, 0.5, 0.5, 0.15, 0.10)
print(f"objects between 0.10 and 0.15 of the centre: {shell.size}")

# Each query also has an overload taking an object already in the grid, which
# is then excluded from its own answer.
around = grid.nearest_objects(5, nearest)
print(f"five nearest to object {nearest}: {around}\n")

# Indices are stable: removing an object leaves every other index valid and its
# own permanently invalid, and adding issues a fresh one. Indices are never
# reused, so a stored index never comes to mean a different object.
grid.remove_object(nearest)
print(f"removed object {nearest}: {len(grid)} objects left, "
      f"is_alive({nearest}) = {grid.is_alive(nearest)}")

added = grid.add_object(0.5, 0.5, 0.5)
print(f"added an object at the centre: index {added}, {len(grid)} objects")
print(f"nearest to the centre is now object {grid.nearest_object(0.5, 0.5, 0.5)}\n")

# Errors are exceptions, never return codes. meshsearch.Error derives from
# ValueError; meshsearch.IndexError derives from it and from Python's
# IndexError, so an index fault answers to either name.
try:
    grid.nearest_object(nearest)
except meshsearch.IndexError as e:
    print(f"querying the removed object raised IndexError: {e}")

try:
    grid.close_objects(2.0, 2.0, 2.0, 0.1)
except meshsearch.Error as e:
    print(f"querying a point outside the box raised Error: {e}")

# Catching the base catches both, as catch (const Error&) does in C++.
for bad in (lambda: grid.nearest_object(10 ** 6),
            lambda: grid.nearest_object(2.0, 2.0, 2.0)):
    try:
        bad()
    except meshsearch.Error as e:
        print(f"caught as meshsearch.Error: {type(e).__name__}")
