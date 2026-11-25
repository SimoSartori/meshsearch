# meshsearch
Author: Simone Sartori (simone.sartori@inaf.it, simosart23@gmail.com)

meshsearch is a C++17 library, with Python bindings, for exact neighbour
searches in three-dimensional point sets. It divides a box around the points
into a uniform mesh of cubic *cells* and answers each *query* by examining
only the cells that can hold a result: the nearest object, the *N* nearest,
every object within a radius or in a spherical shell, and their number. Each
query can be asked about one point or about a batch of points in one call,
and any axis of the box may be periodic.

The library depends only on the C++ standard library, and the Python module
is built on the same code. Documentation for both:
**<https://simosartori.github.io/meshsearch/>**

<!-- docs:install-start -->
## Install

**Python** (3.10 or later; numpy is the only dependency):

```
pip install meshsearch
```

Wheels are published for Linux (x86_64 and aarch64) and macOS (arm64 and
x86_64); elsewhere pip builds the module from source, which needs CMake 3.16
or later and a C++17 compiler. The wheels include free-threaded CPython 3.14t, where the module runs
without re-enabling the GIL.

**C++**, with CMake 3.16 or later. Build and install the library:

```
cmake -S . -B build
cmake --build build
cmake --install build --prefix /path/to/prefix
```

then find it from your own project, configured with
`-DCMAKE_PREFIX_PATH=/path/to/prefix` if the prefix is not a standard one:

```cmake
find_package(meshsearch 0.3 CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE meshsearch::meshsearch)
```

Or build it as part of your project, under the same target name:

```cmake
include(FetchContent)
FetchContent_Declare(meshsearch
  GIT_REPOSITORY https://github.com/SimoSartori/meshsearch.git
  GIT_TAG        v0.3.0)
FetchContent_MakeAvailable(meshsearch)
target_link_libraries(your_target PRIVATE meshsearch::meshsearch)
```

The option `MESHSEARCH_INSTALL` (on by default) determines whether meshsearch adds
its install rules to the build that includes it.
<!-- docs:install-end -->

<!-- docs:example-start -->
## A first grid

In Python, a grid is built from three coordinate arrays, the limits of the box
and a cell side. Objects are identified by their position in the arrays:

```python
import numpy as np
import meshsearch

rng = np.random.default_rng(1)
x, y, z = rng.random((3, 1000))              # 1000 points in the unit cube
grid = meshsearch.MeshGrid(x, y, z, 0.2, limits=[[0, 1], [0, 1], [0, 1]])

grid.nearest_object(0.5, 0.5, 0.5)           # the index of the nearest object
grid.nearest_objects(10, 0.5, 0.5, 0.5)      # the ten nearest, nearest first
grid.close_objects(0.5, 0.5, 0.5, 0.15)      # every object within 0.15
grid.count_objects(0.5, 0.5, 0.5, 0.15)      # how many there are

qx, qy, qz = rng.random((3, 200))            # 200 query points at once
nearest = grid.nearest_object(qx, qy, qz)    # one index per query, in order
```

The same in C++:

```cpp
#include "meshsearch/MeshGrid.h"

#include <array>
#include <cstdio>
#include <random>
#include <vector>

int main ()
{
  std::mt19937 gen(1);
  std::uniform_real_distribution<double> unit(0., 1.);
  std::vector<double> X, Y, Z;
  for (int i=0; i<1000; ++i) {
    X.push_back(unit(gen));
    Y.push_back(unit(gen));
    Z.push_back(unit(gen));
  }

  meshsearch::MeshGrid::Options options;
  options.limits = std::array<std::array<double,2>,3>{{{0.,1.}, {0.,1.}, {0.,1.}}};
  meshsearch::MeshGrid grid(X, Y, Z, 0.2, options);

  const unsigned int nearest = grid.nearestObject(0.5, 0.5, 0.5);
  const std::vector<unsigned int> ten = grid.nearestObjects(10, 0.5, 0.5, 0.5);
  const std::vector<unsigned int> ball = grid.closeObjects(0.5, 0.5, 0.5, 0.15);
  const size_t counted = grid.countObjects(0.5, 0.5, 0.5, 0.15);

  std::printf("nearest %u, %zu objects within 0.15\n", nearest, ball.size());
  return ten.size() == 10 && ball.size() == counted ? 0 : 1;
}
```
<!-- docs:example-end -->

## Where to read next

- [Guide](https://simosartori.github.io/meshsearch/guide.html): the box, the
  cell side, every query, batches, periodic boxes, errors, threads and memory.
- [C++ reference](https://simosartori.github.io/meshsearch/cpp.html) and
  [Python reference](https://simosartori.github.io/meshsearch/python.html).
  `include/meshsearch/MeshGrid.h` is the specification: it declares the
  interface and states every guarantee.
- [Performance](https://simosartori.github.io/meshsearch/benchmark.html):
  timings and memory against scipy's `cKDTree` and scikit-learn's `BallTree`.
- [Changelog](CHANGELOG.md).

## Tests

`ctest --test-dir build` runs the C++ tests and examples; `pytest tests` runs
the Python ones against an installed module. The tests compare every query
with a brute-force search on random points, dense and sparse.

## Licence

BSD 3-Clause; see `LICENSE`.

The code was first developed as part of the
[CosmoBolognaLib](https://github.com/federicomarulli/CosmoBolognaLib)
(Marulli, Veropalumbo & Moresco 2016, A&C, 14, 35).
