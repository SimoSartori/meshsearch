# meshsearch

A uniform cubic mesh over a set of points in three dimensions, for neighbour
queries: the object nearest a point, the *N* nearest, and every object in a
spherical shell. C++17 with no dependencies outside the standard library, and a
Python module built on the same code.

The two APIs are one library with two spellings. Everything the C++ class does,
the Python module does under a snake_case name, with numpy arrays in place of
vectors; the only thing that exists on one side alone is the batched query form,
which Python has and C++ does not need. [Names side by side](naming.md) maps
them one to one.

## Install

```
pip install meshsearch
```

For the C++ library, build it with CMake and link the static library:

```
cmake -S . -B build
cmake --build build
```

## In two lines

**Python**

```python
import numpy as np
import meshsearch

grid = meshsearch.MeshGrid(x, y, z, cellsize)     # three arrays, one cell side

grid.nearest_object(0.5, 0.5, 0.5)                # an index
grid.nearest_objects(10, 0.5, 0.5, 0.5)           # the ten nearest, nearest first
grid.close_objects(0.5, 0.5, 0.5, 0.15, 0.10)     # every object in a shell
```

**C++**

```cpp
#include "meshsearch/MeshGrid.h"

meshsearch::MeshGrid grid(X, Y, Z, cellsize);     // three vectors, one cell side

grid.nearestObject(0.5, 0.5, 0.5);                // an index
grid.nearestObjects(10, 0.5, 0.5, 0.5);           // the ten nearest, nearest first
grid.closeObjects(0.5, 0.5, 0.5, 0.15, 0.10);     // every object in a shell
```

Objects are identified by their position in the arrays passed to the
constructor, and those indices are stable for the life of the grid. The
[guide](guide.md) covers that and the rest of what is worth knowing before the
reference pages.

```{toctree}
:maxdepth: 2
:hidden:

guide
cpp
python
naming
benchmark
```
