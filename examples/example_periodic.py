"""A periodic box with meshsearch: neighbours are found across the faces of the
box, as if it repeated forever in every direction.

    pip install meshsearch
    python examples/example_periodic.py
"""

import numpy as np

import meshsearch

rng = np.random.default_rng(42)
n = 5000
x, y, z = rng.random(n), rng.random(n), rng.random(n)

# Two objects on opposite sides of the x faces, 0.004 apart through the boundary
# and 0.996 apart across the box.
left, right = n, n + 1
x = np.append(x, [0.002, 0.998])
y = np.append(y, [0.5, 0.5])
z = np.append(z, [0.5, 0.5])

# A periodic axis needs its limits: they are the box, and the period is hi-lo.
# Coordinates are not wrapped, so every point must already lie within them.
# The periods must be whole multiples of the shortest one. periodic is a bool
# for all three axes or one bool per axis, so a slab periodic in x and y only
# would be periodic=(True, True, False). Both are keyword-only.
cellsize = 2 * (1.0 / n) ** (1 / 3)
grid = meshsearch.MeshGrid(x, y, z, cellsize,
                           limits=[[0, 1], [0, 1], [0, 1]], periodic=True)

# The cell side used is the largest that fits a whole number of times into
# the shortest period without exceeding the one asked for, so read it back.
print(f"periodic {grid.periodic}, cellsize asked {cellsize:.6f}, "
      f"used {grid.cellsize:.6f}, {'x'.join(str(c) for c in grid.n_cells)} cells")

# The nearest object to `left` is `right`, through the x boundary.
nearest = grid.nearest_object(left)
print(f"nearest to object {left}: object {nearest} (expected {right})")

# Radii are not limited by the box: +inf returns every object once, each at its
# nearest image.
everything = grid.close_objects(0.5, 0.5, 0.5, np.inf)
print(f"objects within +inf of the centre: {everything.size} of {len(grid)}")

# A ball around a point at the corner takes in objects from all eight corners
# of the box.
corner = grid.close_objects(0.0, 0.0, 0.0, 0.1)
print(f"objects within 0.1 of the corner (0, 0, 0): {corner.size}")

assert nearest == right and everything.size == len(grid)
