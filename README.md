# meshsearch
Author: Simone Sartori (simone.sartori@inaf.it, simosart23@gmail.com)

A uniform cubic mesh over a set of points in three dimensions, for neighbour
queries: the object nearest a point, the N nearest, and every object in a
spherical shell. Objects are indexed by their position in the coordinate
vectors passed to the constructor, and can be added and removed afterwards.

C++17, no dependencies outside the standard library. Extracted from
CosmoBolognaLib, but it depends on nothing from it.

`include/meshsearch/MeshGrid.h` is the specification: it declares the
interface and states the guarantees it makes.

## Build

Needs CMake 3.16 or later and a C++17 compiler.

```
cmake -S . -B build
cmake --build build
```

That produces the static library `libmeshsearch.a`, the test binary and the
example. CI configures, builds and tests on Ubuntu and macOS on every push;
locally it is also built with GCC 13, 14, 15 and 16, and with AppleClang 15.

A version of this code is distributed as part of the
[CosmoBolognaLib](https://github.com/federicomarulli/CosmoBolognaLib)
(Marulli, Veropalumbo & Moresco 2016, A&C, 14, 35), where it was
originally developed.

## Tests

```
ctest --test-dir build
```

The load-bearing tests compare every query against brute force over several
thousand random points, in both dense and sparse regimes, and with duplicate
and coincident points. The rest check the guarantees the header states.

## Example

`examples/example.cpp` is a short tour of the interface; run it with
`./build/example`. In outline:

```cpp
#include "meshsearch/MeshGrid.h"

// coordinates as three parallel vectors, and a cell side in the same unit
meshsearch::MeshGrid grid(X, Y, Z, cellsize);

unsigned int nearest = grid.nearestObject(0.5, 0.5, 0.5);
std::vector<unsigned int> ten = grid.nearestObjects(10, 0.5, 0.5, 0.5);
std::vector<unsigned int> shell = grid.closeObjects(0.5, 0.5, 0.5, 0.15, 0.10);

// the same questions about an object in the grid, which is excluded from
// its own answer
std::vector<unsigned int> around = grid.nearestObjects(5, nearest);
```

Three things to know before using it. A shell is `Rmin <= d <= Rmax`, closed at
both ends, so a ball holds every object within `Rmax` and an object sitting
exactly on either radius is returned. Indices are stable:
`removeObject` leaves an index permanently invalid, `addObject` issues a new
one, and indices are never reused or renumbered, so a stored index never comes
to mean a different object. And errors are exceptions — `meshsearch::Error`, or
`meshsearch::IndexError` for an index that is out of range or removed — never
return codes.

The overloads taking an object exclude it **by identity, not by distance**:
`closeObjects(i, Rmax)` omits object `i` itself, but another object at the same
position is a different object, is at `d = 0`, and is returned. The same holds
for `nearestObject` and `nearestObjects`. The point-taking overloads have no
identity to exclude, so an object lying on the query point is simply within the
radius and is returned.

Const members may be called concurrently on one grid; non-const members may
not.

## Python

The same grid is available from Python, with snake_case names and numpy
results. Install from the source tree:

```
pip install .
```

```python
import numpy as np
import meshsearch

grid = meshsearch.MeshGrid(x, y, z, cellsize)          # three arrays, one cell side

len(grid)                                              # live object count
grid.lims, grid.n_cells, grid.cellsize                 # read-only properties

grid.nearest_object(0.5, 0.5, 0.5)                     # an index
grid.nearest_objects(10, 0.5, 0.5, 0.5)                # uint32 array, nearest first
grid.close_objects(0.5, 0.5, 0.5, 0.15, 0.10)          # uint32 array, a shell
grid.nearest_objects(5, index)                         # about an object, which is excluded
```

`meshsearch.Error` derives from `ValueError`. `meshsearch.IndexError` derives
from both `meshsearch.Error` and Python's `IndexError`, mirroring the C++
hierarchy, so an index fault can be caught either as the Python type it stands
for or along with every other error the grid raises.

Run the Python tests with `pytest tests/test_meshgrid.py` against an installed
build.

### Batched queries

The three point-taking queries also accept arrays of query points, answering
all of them in one call:

```python
nearest = grid.nearest_object(qx, qy, qz)              # (n_query,) uint32
five    = grid.nearest_objects(5, qx, qy, qz)          # (n_query, 5) uint32
idx, off = grid.close_objects(qx, qy, qz, 0.02)        # compressed-row form
```

The same names are overloaded: three scalars ask about one point, three arrays
ask about many. Query points must be numpy arrays rather than lists, because an
array argument is what selects the batched form. A one-element array takes the
batched path and returns a one-element result.

**This exists only in Python, and that is deliberate.** Crossing the Python
boundary costs a few microseconds per call, which is more than some of these
queries take to run: it is why calling scipy's `cKDTree` once per point is
about five times slower than handing it the whole array, on identical work. A
C++ caller has no such boundary and writes a loop, so the C++ library has no
batched API and is not missing one. The **index-taking overloads have no
batched form either**, in Python or C++: their input is a single index, so
there is no per-call cost to amortise over anything.

A batched call raises where the single-point call would, naming the query that
failed — `query 17: X coordinate outside the box` — and returns nothing
partial.

### The compressed-row result

`nearest_objects` is rectangular: every query returns exactly `n`, so the
result is one `(n_query, n)` array. A shell is not — the count varies per query
— so `close_objects` returns two arrays instead:

- **`indices`** holds every result, concatenated in query order.
- **`offsets`** has `n_query + 1` entries. The results of query `i` are
  `indices[offsets[i]:offsets[i+1]]`. `offsets[0]` is `0` and the last entry is
  `len(indices)`. A query with no results gives an empty slice, never a missing
  entry. Both arrays are `uint32`, and `offsets` counts results, not queries, so
  it is wide enough for the whole batch.

```python
idx, off = grid.close_objects(qx, qy, qz, 0.02)

idx[off[i]:off[i+1]]          # the neighbours of query i, no copy
counts = np.diff(off)         # how many each query found
counts.sum() == len(idx)      # always
```

The idiom that makes this form worth learning is pairing each result with the
query it came from:

```python
query_of = np.repeat(np.arange(len(qx)), np.diff(off))
```

`indices` and `query_of` are now two flat arrays of the same length, so any
per-query quantity is one vectorised expression over every result at once —
`np.bincount(query_of, weights=...)` to reduce per query, boolean masks to
select across queries. `examples/example_batch.py` works through it.

**Why not a list of arrays?** Because that would allocate one Python object per
query, which is the cost the batched call exists to remove: a batch of 50,000
queries would hand back 50,000 numpy arrays. There is deliberately no helper to
convert the compressed form into such a list, it would be slow and would undo
the point. Slice it, or use the flat form directly.

## Licence

BSD 3-Clause; see `LICENSE`.
