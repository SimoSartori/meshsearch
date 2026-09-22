# Names side by side

The same members, spelled for each language. C++ uses the names the header
declares; Python uses snake_case, turns the three accessors into properties,
and returns numpy arrays where C++ returns vectors.

| C++ | Python | notes |
|---|---|---|
| `MeshGrid(X, Y, Z, cellsize, limits)` | `MeshGrid(x, y, z, cellsize, limits)` | `limits` optional in both |
| `get_nObjects()` | `len(grid)` | live objects, removals excluded |
| `isAlive(index)` | `is_alive(index)` | |
| `get_lims()` | `grid.lims` | property; a 3×2 array in Python |
| `get_nCells()` | `grid.n_cells` | property; a tuple in Python |
| `get_cellsize()` | `grid.cellsize` | property |
| `nearestObject(X, Y, Z)` | `nearest_object(x, y, z)` | |
| `nearestObject(index)` | `nearest_object(index)` | the object itself excluded |
| `nearestObjects(N, X, Y, Z)` | `nearest_objects(n, x, y, z)` | |
| `nearestObjects(N, index)` | `nearest_objects(n, index)` | |
| `closeObjects(X, Y, Z, Rmax, Rmin)` | `close_objects(x, y, z, rmax, rmin)` | `Rmin` defaults to 0 |
| `closeObjects(index, Rmax, Rmin)` | `close_objects(index, rmax, rmin)` | |
| `get_Cell(X, Y, Z)` | `get_cell(x, y, z)` | |
| `get_CellCoords(X, Y, Z)` | `get_cell_coords(x, y, z)` | |
| `get_ObjectsInCell(X, Y, Z)` | `get_objects_in_cell(x, y, z)` | |
| `addObject(X, Y, Z)` | `add_object(x, y, z)` | returns the new index |
| `removeObject(index)` | `remove_object(index)` | |
| `meshsearch::Error` | `meshsearch.Error` | derives from `ValueError` in Python |
| `meshsearch::IndexError` | `meshsearch.IndexError` | derives from both `meshsearch.Error` and the built-in `IndexError` |

## What exists on one side only

**Batched queries, Python only.** The three point-taking queries also accept
arrays of query points and answer them in one call. C++ has no equivalent and
needs none: the cost batching removes is crossing the Python boundary, and a C++
caller writes a loop. The [guide](guide.md#batched-queries-and-the-form-they-return)
covers the form they return.

**The index-taking overloads have no batched form in either language,** their
input being a single index.

## Where the two behave differently

Nowhere, deliberately. The Python module is a thin binding: same guarantees,
same exceptions, same order of results, same closed interval `Rmin <= d <= Rmax`
on `closeObjects` / `close_objects`. What differs is only the shape of what
comes back — a numpy `uint32` array in place of a `std::vector<unsigned int>`,
a tuple in place of a `std::array` — and that the Python arrays own their data
and are safe to keep after the grid is gone.
