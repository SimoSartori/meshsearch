# meshsearch

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
example. Verified on macOS with AppleClang 15 and with GCC 13, 14, 15 and 16;
the code is portable to Linux but has not been run there.

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

## Licence

BSD 3-Clause; see `LICENSE`.
