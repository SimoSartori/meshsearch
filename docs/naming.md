# Names side by side

The same members, spelled for each language. C++ uses the names the header
declares; Python uses snake_case, turns the three accessors into properties,
and returns numpy arrays where C++ returns vectors.

| C++ | Python | notes |
|---|---|---|
| `MeshGrid(X, Y, Z, cellsize, options)` | `MeshGrid(x, y, z, cellsize, *, limits=None, periodic=False)` | C++ passes `limits` and `periodic` in a `MeshGrid::Options`; Python takes them as keywords |
| `get_periodic()` | `grid.periodic` | property; a tuple of three bools in Python |
| `get_nObjects()` | `len(grid)` | live objects, removals excluded |
| `isAlive(index)` | `is_alive(index)` | the one member taking an index that does not raise |
| `get_lims()` | `grid.lims` | property; `std::array<std::array<double,2>,3>` in C++, a 3×2 array in Python |
| `get_nCells()` | `grid.n_cells` | property; a tuple in Python |
| `get_cellsize()` | `grid.cellsize` | property |
| `nearestObject(X, Y, Z)` | `nearest_object(x, y, z)` | |
| `nearestObject(index)` | `nearest_object(index)` | the object itself excluded |
| `nearestObjects(N, X, Y, Z)` | `nearest_objects(n, x, y, z)` | |
| `nearestObjects(N, index)` | `nearest_objects(n, index)` | |
| `closeObjects(X, Y, Z, Rmax, Rmin)` | `close_objects(x, y, z, rmax, rmin)` | `Rmin` defaults to 0 |
| `closeObjects(index, Rmax, Rmin)` | `close_objects(index, rmax, rmin)` | |
| `countObjects(X, Y, Z, Rmax, Rmin)` | `count_objects(x, y, z, rmax, rmin)` | a `std::size_t`; an `int` in Python |
| `countObjects(index, Rmax, Rmin)` | `count_objects(index, rmax, rmin)` | |
| `nearestObject(QX, QY, QZ)` | `nearest_object(qx, qy, qz, *, out=None)` | batch: vectors in C++, numpy arrays in Python |
| `nearestObjects(N, QX, QY, QZ)` | `nearest_objects(n, qx, qy, qz, *, out=None)` | `n*N` flat in C++; `(n_query, n)` in Python |
| `closeObjects(QX, QY, QZ, Rmax, Rmin)` | `close_objects(qx, qy, qz, rmax, rmin)` | a `MeshGrid::Batch`; a tuple `(indices, offsets)` in Python, offsets `int64` |
| `countObjects(QX, QY, QZ, Rmax, Rmin)` | `count_objects(qx, qy, qz, rmax, rmin, *, out=None)` | `std::vector<std::uint64_t>`; `int64` in Python |
| `nearestObject(indices)` | `nearest_object(indices, *, out=None)` | batch over objects; Python takes any integer array |
| `nearestObjects(N, indices)` | `nearest_objects(n, indices, *, out=None)` | |
| `closeObjects(indices, Rmax, Rmin)` | `close_objects(indices, rmax, rmin)` | |
| `countObjects(indices, Rmax, Rmin)` | `count_objects(indices, rmax, rmin, *, out=None)` | |
| `MeshGrid::Batch` | — | `indices` and `offsets` members; Python returns the two arrays |
| `get_cell(X, Y, Z)` | `get_cell(x, y, z)` | |
| `get_cellCoords(X, Y, Z)` | `get_cell_coords(x, y, z)` | |
| `get_objectsInCell(X, Y, Z)` | `get_objects_in_cell(x, y, z)` | |
| `addObject(X, Y, Z)` | `add_object(x, y, z)` | returns the new index |
| `removeObject(index)` | `remove_object(index)` | |
| `meshsearch::Error` | `meshsearch.Error` | derives from `ValueError` in Python |
| `meshsearch::IndexError` | `meshsearch.IndexError` | derives from both `meshsearch.Error` and the built-in `IndexError` |

## What exists on one side only

**Container forms, C++ only.** Every C++ query that returns a container has an
overload taking one as its last argument, which is cleared, filled, and keeps
its capacity: `closeObjects(X, Y, Z, Rmax, Rmin, out)`,
`nearestObjects(N, index, out)`, `countObjects(QX, QY, QZ, Rmax, out)` and so
on. The [guide](guide.md#reusing-a-container-in-c) shows them in a loop.

**`out=`, Python only.** The batched `nearest_object`, `nearest_objects` and `count_objects` take
a keyword-only `out=`, an array to fill and return instead of allocating one.
The [guide](guide.md#reusing-an-array-in-python) gives its rules.

## Where the two behave differently

Nowhere, deliberately. The Python module is a thin binding: same guarantees,
same exceptions, same order of results, same closed interval `Rmin <= d <= Rmax`
on `closeObjects` / `close_objects`. What differs is only the shape of what
comes back — a numpy `uint32` array in place of a `std::vector<unsigned int>`,
an `int64` array in place of a `std::vector<std::uint64_t>`, a tuple in place
of a `std::array` or a `MeshGrid::Batch` — and that the Python arrays own their
data and are safe to keep after the grid is gone. A failed Python batch also
names the query that failed in its message.
