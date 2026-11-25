"""A short tour of meshsearch from Python: build a grid over a set of points,
ask the four neighbour queries, modify the grid, and handle a bad query.

    pip install meshsearch
    python examples/example.py
"""

import numpy as np

import meshsearch

# The points, as three arrays of the same length. An object is identified by
# its position in them, and the arrays are copied into the grid.
rng = np.random.default_rng(42)
n = 5000
x, y, z = rng.random(n), rng.random(n), rng.random(n)

# The box. Give the limits of the region the points and every later query
# point belong to, here the unit cube. Without limits the box is fitted to the
# points, and a query point outside their range raises meshsearch.Error.
limits = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

# The cell side, the one tuning parameter. The mean separation of n points in
# a volume V is (V / n) ** (1/3). On uniform data the guide advises starting
# between 1.5 and 2.5 mean separations, here two, and shows what other choices
# cost. Coordinates, cell side and radii share one unit, whatever it is.
volume = 1.0
separation = (volume / n) ** (1 / 3)
cellsize = 2 * separation

grid = meshsearch.MeshGrid(x, y, z, cellsize, limits=limits)

print(f"{len(grid)} objects, {'x'.join(str(c) for c in grid.n_cells)} cells "
      f"of {grid.cellsize:.4f}")
for axis, (lo, hi) in zip("xyz", grid.lims):
    print(f"  {axis} [{lo:.4f}, {hi:.4f}]")
print()

nearest = grid.nearest_object(0.5, 0.5, 0.5)
print(f"nearest to the box centre: object {nearest} at "
      f"({x[nearest]:.4f}, {y[nearest]:.4f}, {z[nearest]:.4f})")

# Index results are uint32 arrays, so they index straight back into the
# coordinate arrays.
ten = grid.nearest_objects(10, 0.5, 0.5, 0.5)
print(f"ten nearest to the centre: {ten}")
print("their distances:",
      np.round(np.sqrt((x[ten] - 0.5) ** 2 + (y[ten] - 0.5) ** 2
                       + (z[ten] - 0.5) ** 2), 4))

# A shell is rmin <= d <= rmax, both ends included; with rmin left out it is a
# ball. The order of the indices is not meaningful.
shell = grid.close_objects(0.5, 0.5, 0.5, 0.15, 0.10)
counted = grid.count_objects(0.5, 0.5, 0.5, 0.15, 0.10)
print(f"objects between 0.10 and 0.15 of the centre: {shell.size} (counted: {counted})")

# Asked about an object in the grid, a query leaves that object out of its own
# answer.
around = grid.nearest_objects(5, nearest)
print(f"five nearest to object {nearest}: {around}\n")

# Removing an object leaves its index permanently invalid; adding one issues a
# new index. No index is ever reused. Neither call may run while another
# thread uses the grid.
grid.remove_object(nearest)
print(f"removed object {nearest}: {len(grid)} objects left, "
      f"is_alive({nearest}) = {grid.is_alive(nearest)}")

added = grid.add_object(0.5, 0.5, 0.5)
print(f"added an object at the centre: index {added}, {len(grid)} objects")
print(f"nearest to the centre is now object {grid.nearest_object(0.5, 0.5, 0.5)}\n")

# meshsearch.IndexError derives from meshsearch.Error and from Python's
# IndexError, so an index fault answers to either name.
try:
    grid.nearest_object(nearest)
except meshsearch.IndexError as e:
    print(f"querying the removed object raised IndexError: {e}")

# A point outside the box is an error, never a silent empty answer; contains()
# tells beforehand.
print(f"contains(2, 2, 2) = {grid.contains(2.0, 2.0, 2.0)}")
try:
    grid.close_objects(2.0, 2.0, 2.0, 0.1)
except meshsearch.Error as e:
    print(f"querying a point outside the box raised Error: {e}")

assert shell.size == counted and ten.size == 10
