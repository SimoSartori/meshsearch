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

Two things to know before using it. Indices are stable: `removeObject` leaves
an index permanently invalid, `addObject` issues a new one, and indices are
never reused or renumbered, so a stored index never comes to mean a different
object. And errors are exceptions — `meshsearch::Error`, or
`meshsearch::IndexError` for an index that is out of range or removed — never
return codes.

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

## Licence

BSD 3-Clause; see `LICENSE`.
