# Names side by side

The same members, spelled for each language. C++ uses the names the header
declares; Python uses snake_case, makes the four accessors `periodic`, `lims`,
`n_cells` and `cellsize` read-only properties, gives the object count as
`len(grid)`, and returns numpy arrays where C++ returns vectors. Each member
is documented in the [C++ reference](cpp.rst) and the
[Python reference](python.rst).

In the table, `X, Y, Z` are the coordinates of one point and `QX, QY, QZ` the
coordinate arrays of a batch of query points; `index` is one object index and
`indices` an array of them.

| C++ | Python | what it is |
|---|---|---|
| `MeshGrid(X, Y, Z, cellsize)`, `MeshGrid(X, Y, Z, cellsize, options)` | `MeshGrid(x, y, z, cellsize, *, limits=None, periodic=False)` | C++ passes limits and periodic axes in a `MeshGrid::Options` |
| `MeshGrid(pX, pY, pZ, n, cellsize)`, `MeshGrid(pX, pY, pZ, n, cellsize, options)` | — | the same from three pointers to `n` coordinates each |
| `MeshGrid(pXYZ, n, cellsize)`, `MeshGrid(pXYZ, n, cellsize, options)` | `MeshGrid(pos, cellsize, *, limits=None, periodic=False)` | the same from `n` points as rows x, y, z: one pointer to `3n` doubles, an `(N, 3)` array |
| `MeshGrid()` | — | an empty grid, to assign to later |
| `get_nObjects()` | `len(grid)` | objects in the grid, removed ones excluded |
| `get_lims()` | `grid.lims` | the box; a 3×2 array in Python |
| `get_nCells()` | `grid.n_cells` | cells per axis; a tuple in Python |
| `get_cellsize()` | `grid.cellsize` | the cell side used |
| `get_periodic()` | `grid.periodic` | periodic flag per axis; a tuple in Python |
| `isAlive(index)`, `isAlive(indices)` | `is_alive(index)`, `is_alive(indices, *, out=None)` | whether an index names an object in the grid |
| `contains(X, Y, Z)`, `contains(QX, QY, QZ)` | `contains(x, y, z)`, `contains(qx, qy, qz, *, out=None)` | whether a point lies in the box |
| `nearestObject(X, Y, Z)`, `nearestObject(index)` | `nearest_object(x, y, z)`, `nearest_object(index)` | the nearest object |
| `nearestObject(QX, QY, QZ)`, `nearestObject(indices)` | `nearest_object(qx, qy, qz, *, out=None)`, `nearest_object(indices, *, out=None)` | its batches |
| `nearestObjects(N, X, Y, Z)`, `nearestObjects(N, index)` | `nearest_objects(n, x, y, z)`, `nearest_objects(n, index)` | the *N* nearest |
| `nearestObjects(N, QX, QY, QZ)`, `nearestObjects(N, indices)` | `nearest_objects(n, qx, qy, qz, *, out=None)`, `nearest_objects(n, indices, *, out=None)` | its batches: `n*N` flat in C++, `(n_query, n)` in Python |
| `closeObjects(X, Y, Z, Rmax, Rmin)`, `closeObjects(index, Rmax, Rmin)` | `close_objects(x, y, z, rmax, rmin=0)`, `close_objects(index, rmax, rmin=0)` | every object in a ball or shell |
| `closeObjects(QX, QY, QZ, Rmax, Rmin)`, `closeObjects(indices, Rmax, Rmin)` | `close_objects(qx, qy, qz, rmax, rmin=0)`, `close_objects(indices, rmax, rmin=0)` | its batches, returning a `Batch` |
| `countObjects(X, Y, Z, Rmax, Rmin)`, `countObjects(index, Rmax, Rmin)` | `count_objects(x, y, z, rmax, rmin=0)`, `count_objects(index, rmax, rmin=0)` | how many `closeObjects` would return |
| `countObjects(QX, QY, QZ, Rmax, Rmin)`, `countObjects(indices, Rmax, Rmin)` | `count_objects(qx, qy, qz, rmax, rmin=0, *, out=None)`, `count_objects(indices, rmax, rmin=0, *, out=None)` | its batches |
| `MeshGrid::Batch` | `meshsearch.Batch` | `indices` and `offsets`; a named tuple in Python |
| `get_cell(X, Y, Z)` | `get_cell(x, y, z)` | linear index of a point's cell |
| `get_cellCoords(X, Y, Z)` | `get_cell_coords(x, y, z)` | coordinates of a point's cell |
| `get_objectsInCell(X, Y, Z)` | `get_objects_in_cell(x, y, z)` | the objects in a point's cell |
| `addObject(X, Y, Z)` | `add_object(x, y, z)` | add an object; returns its index |
| `removeObject(index)` | `remove_object(index)` | remove an object |
| copy constructor and assignment | `copy.copy(grid)`, `copy.deepcopy(grid)` | an independent copy |
| move constructor and assignment | — | transfer a grid without copying |
| — | `repr(grid)` | object count, cells, cell side and periodic axes |
| `MESHSEARCH_VERSION` in `meshsearch/Version.h` | `meshsearch.__version__` | the version built |
| `meshsearch::Error` | `meshsearch.Error` | every error the grid raises; a `ValueError` in Python |
| `meshsearch::IndexError` | `meshsearch.IndexError` | an index out of range or removed; also a built-in `IndexError` in Python |
| `std::bad_alloc` | `MemoryError` | out of memory |

## What differs

The results and the errors are the same in both languages; only their types
differ.

- **Types.** Indices are a `std::vector<unsigned int>` in C++ and a `uint32`
  array in Python; batch counts a `std::vector<std::uint64_t>` and an `int64`
  array; flags a `std::vector<std::uint8_t>` of 0 and 1 and a `bool` array;
  batch offsets `std::uint64_t` and `int64`. Python arrays own their data and
  stay valid after the grid is gone.
- **Reusing memory.** C++ queries that return a container (all but
  `get_objectsInCell`) also have a form writing into one passed last; Python
  batches with a result of known shape take `out=`. See the
  [guide](guide.md#reusing-memory-between-calls).
- **Negative numbers.** The unsigned C++ types cannot express them; in Python
  a negative index raises `meshsearch.IndexError` (and `is_alive` answers
  `False`), and a negative `n` raises `meshsearch.Error`.
- **Messages.** A failed Python batch names the query that failed.
