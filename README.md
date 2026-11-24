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

Documentation, covering both APIs in one place:
**<https://simosartori.github.io/meshsearch/>**

## Build

Needs CMake 3.16 or later and a C++17 compiler.

```
cmake -S . -B build
cmake --build build
```

That produces the static library `libmeshsearch.a`, the test binary and the
example. CI configures, builds and tests on Ubuntu and macOS on every push;
locally it is also built with GCC 13, 14, 15 and 16, and with AppleClang 15.

The test binary and the example are built only when meshsearch is the top-level
project, not when it is included by another one.

### Using it from CMake

Install it, then find it as a package:

```
cmake --install build --prefix /path/to/prefix
```

```cmake
find_package(meshsearch 0.2 CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE meshsearch::meshsearch)
```

Configure your project with `-DCMAKE_PREFIX_PATH=/path/to/prefix` if the prefix
is not a standard one. Or build it as part of your project, with the same
target name:

```cmake
include(FetchContent)
FetchContent_Declare(meshsearch
  GIT_REPOSITORY https://github.com/SimoSartori/meshsearch.git
  GIT_TAG        v0.2.1)
FetchContent_MakeAvailable(meshsearch)
target_link_libraries(your_target PRIVATE meshsearch::meshsearch)
```

`MESHSEARCH_INSTALL` (on by default) controls whether meshsearch adds its
install rules to the build that includes it.

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

#include <array>
#include <cstdio>
#include <random>
#include <vector>

int main ()
{
  // coordinates as three parallel vectors, and a cell side in the same unit
  std::mt19937 gen(1);
  std::uniform_real_distribution<double> unit(0., 1.);
  std::vector<double> X, Y, Z;
  for (int i=0; i<1000; ++i) {
    X.push_back(unit(gen));
    Y.push_back(unit(gen));
    Z.push_back(unit(gen));
  }
  const double cellsize = 0.125;
  meshsearch::MeshGrid grid(X, Y, Z, cellsize);

  // a box with limits, periodic in every direction
  meshsearch::MeshGrid::Options options;
  options.limits = std::array<std::array<double,2>,3>{{{0.,1.}, {0.,1.}, {0.,1.}}};
  options.periodic = {true, true, true};
  meshsearch::MeshGrid box(X, Y, Z, cellsize, options);

  unsigned int nearest = grid.nearestObject(0.5, 0.5, 0.5);
  std::vector<unsigned int> ten = grid.nearestObjects(10, 0.5, 0.5, 0.5);
  std::vector<unsigned int> shell = grid.closeObjects(0.5, 0.5, 0.5, 0.15, 0.10);

  // the same questions about an object in the grid, which is excluded from
  // its own answer
  std::vector<unsigned int> around = grid.nearestObjects(5, nearest);

  std::printf("%zu nearest, %zu in the shell, %zu around object %u\n",
              ten.size(), shell.size(), around.size(), nearest);
  return ten.size() == 10 && around.size() == 5 && box.get_nObjects() == 1000 ? 0 : 1;
}
```

<!-- docs:concepts-start -->
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

The constructor checks its input before allocating anything. It raises
`meshsearch::Error` for a cellsize that is not finite and positive, for limits
that are not finite or have `lo > hi`, for a coordinate that is not finite, for
a cellsize so small that its reciprocal overflows a double, for a box so large
that the square of its diagonal overflows a double (an extent of about 1e154
along an axis), for a grid too large to count: more than 2147483647 cells
along an axis, or more than 4294967295 objects, and for a periodic axis without
limits or whose period is not an integer multiple of the shortest. Running out of memory is not
translated: it is `std::bad_alloc` in C++ and `MemoryError` in Python, and an
`addObject` that fails that way leaves the grid as it was.

Const members may be called concurrently on one grid; non-const members may
not.
<!-- docs:concepts-end -->

## Performance

`docs/benchmark.md` measures meshsearch against scipy's `cKDTree` and sklearn's
`BallTree`, and measures what the cell side costs. At 1e7 uniform points:

- construction is 2x to 3.7x faster than cKDTree;
- nearest neighbour is 5x to 8x faster per call and 2.4x to 3.1x faster
  batched, at a cell side of one mean separation;
- a radius query on a coarse grid is the fastest thing measured, 8.76
  microseconds batched against cKDTree's 9.22;
- `k = 10` is 2x to 7x slower than batched cKDTree — take that query elsewhere
  if it dominates;
- memory is the price: 1.5 GiB at one mean separation against cKDTree's
  400 MiB, falling to 672 MiB at four.

The cell side is the one tuning parameter. Two to three mean separations is a
reasonable default, one mean separation favours nearest-neighbour queries, and
three to four favours shells; `python benchmarks/benchmark_cellsize.py --sweep`
repeats the sweep on your own data.

## Python

The same grid is available from Python, with snake_case names and numpy
results. Install from the source tree:

```
pip install .
```

```python
import numpy as np
import meshsearch

rng = np.random.default_rng(1)
x, y, z = rng.random((3, 1000))                        # the points, in the unit cube
cellsize = 0.125
index = 0                                              # the position of an object

grid = meshsearch.MeshGrid(x, y, z, cellsize)          # three arrays, one cell side
box = meshsearch.MeshGrid(x, y, z, cellsize,           # limits and periodic are
                          limits=[[0, 1]] * 3,         # keyword-only; periodic is
                          periodic=True)               # a bool or one bool per axis

len(grid)                                              # live object count
grid.lims, grid.n_cells, grid.cellsize                 # read-only properties
grid.periodic                                          # (False, False, False)
grid.is_alive(index), grid.contains(0.5, 0.5, 0.5)     # bools; neither raises
meshsearch.__version__                                 # the version of the library

grid.nearest_object(0.5, 0.5, 0.5)                     # an index
grid.nearest_objects(10, 0.5, 0.5, 0.5)                # uint32 array, nearest first
grid.close_objects(0.5, 0.5, 0.5, 0.15, 0.10)          # uint32 array, a shell
grid.nearest_objects(5, index)                         # about an object, which is excluded
```

`meshsearch.Error` derives from `ValueError`. `meshsearch.IndexError` derives
from both `meshsearch.Error` and Python's `IndexError`, mirroring the C++
hierarchy, so an index fault can be caught either as the Python type it stands
for or along with every other error the grid raises. A negative index is out of
range, not counted from the end: it raises `meshsearch.IndexError`, and a
negative `n` raises `meshsearch.Error`. `is_alive` is the one member taking an
index that does not raise; it answers `False`.

`copy.copy(grid)` and `copy.deepcopy(grid)` give an independent grid that
answers every query as the original does. `repr(grid)` shows the object count,
cells, cell side and periodic axes.

Run the Python tests with `pytest tests/test_meshgrid.py` against an installed
build.

### Batched queries

The point-taking queries also accept arrays of query points and answer all of
them in one call, results in query order:

```python
import numpy as np
import meshsearch

rng = np.random.default_rng(1)
x, y, z = rng.random((3, 1000))
grid = meshsearch.MeshGrid(x, y, z, 0.125, limits=[[0, 1]] * 3)
qx, qy, qz = rng.random((3, 200))                      # the query points
indices = np.arange(0, 1000, 5)                        # object indices

nearest = grid.nearest_object(qx, qy, qz)              # (n_query,) uint32
five    = grid.nearest_objects(5, qx, qy, qz)          # (n_query, 5) uint32
idx, off = grid.close_objects(qx, qy, qz, 0.02)        # a meshsearch.Batch
counts  = grid.count_objects(qx, qy, qz, 0.02)         # (n_query,) int64
inside  = grid.contains(qx, qy, qz)                    # (n_query,) bool
alive   = grid.is_alive(indices)                       # (n_query,) bool
```

Any array-like will do for the query points or indices: a float64 or integer
array is read as it is, without a copy, and a list or a float32 array is
converted first. What selects the batch is an argument that has a length, in
place of a number. `close_objects` returns a named tuple `meshsearch.Batch`
`(indices, offsets)`: every index found in one `uint32` array and `int64`
offsets with `n_query + 1` entries, so that query `i` found
`idx[off[i]:off[i+1]]`. `count_objects` counts without collecting. Every query
also takes an array of object indices in place of a single index; an index
that is negative or not valid raises `meshsearch.IndexError`, naming the
query. The batched `nearest_object`, `nearest_objects`, `count_objects`,
`contains` and `is_alive` accept `out=`, an array of exactly the result's dtype
and shape to fill instead of allocating. A batched call releases the GIL while
it runs: other threads may query the same grid meanwhile, but must not modify
it.

C++ has the same batches, under the same names, taking vectors, plus a form of
every query that writes into a container the caller passes last and reuses.
`contains` and the batch `isAlive` return one `std::uint8_t`, 0 or 1, per entry.
`meshsearch/Version.h`, generated by CMake, defines `MESHSEARCH_VERSION`.
The [guide](https://simosartori.github.io/meshsearch/guide.html) covers both.

## Licence

BSD 3-Clause; see `LICENSE`.
