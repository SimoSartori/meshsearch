# Guide

The concepts of the library and their use, in the order a new user needs
them. Everything applies to both languages unless stated otherwise; the
[names side by side](naming.md) give each member's spelling in the other.

## Points, indices and the box

A grid is built from three coordinate arrays of equal length (vectors in C++,
any array-like in Python), a cell side, and optionally the limits of the box.
The coordinates are copied into the grid. The library prescribes no unit:
coordinates, the cell side and every radius must be in the same one.

An *object* is a point held by the grid, and its *index* is its position in
the arrays passed to the constructor. Indices are stable for the life of the
grid: removing an object leaves its index permanently invalid, adding one
issues a new index, and no index is reused or renumbered.

The *box* is the region in which the grid accepts points. Every member that
takes a point, a query point or a new object, raises `Error` for a point
outside it. It is set in one of two ways.

- **Explicit limits**, `{{xlo, xhi}, {ylo, yhi}, {zlo, zhi}}`, describe the
  region to which the points and every later query point belong: the survey
  volume, the simulation box, the unit cube. Every coordinate passed to the
  constructor must lie within them.
- **Fitted limits**, when none are given, are the range of the points,
  widened by a small margin.

**Pass limits whenever a query point can fall outside the range of the
objects.** With fitted limits, a query point beyond the outermost object, or
drawn from a different sample than the objects, can raise `Error`. Whether it
does depends on how the box was rounded to whole cells, which is not part of
the interface. Fitted limits are safe only when every query point is an
object or is known to lie within their range. An empty set of points requires
explicit limits.

```{literalinclude} ../examples/example.py
:language: python
:start-at: "limits = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]"
:end-at: "grid = meshsearch.MeshGrid(x, y, z, cellsize, limits=limits)"
```

```{literalinclude} ../examples/example.cpp
:language: cpp
:start-at: "meshsearch::MeshGrid::Options options;"
:end-at: "meshsearch::MeshGrid grid(X, Y, Z, cellsize, options);"
```

The box is then extended to a whole number of cells. On each axis that is not
periodic it is widened symmetrically, so `get_lims()` (`lims` in Python) can
be slightly larger than the limits given, and always contains them; a point
in the added margin is accepted. An axis along which the box has no extent,
because every point has the same coordinate or the limits have `lo == hi`,
gets a single cell, so flat and one-dimensional point sets are accepted. On a
periodic axis the box is exactly the limits; see
[periodic boxes](#periodic-boxes).

## Choosing the cell side

The cell side is the one tuning parameter. A query visits the cells that can
hold a result and tests every object in them: large cells hold many objects
that are too far away, small cells make a query visit many empty cells and
cost memory.
The cell side affects only speed and memory, never results.

The scale against which to set it is the *mean separation* *s*: for *N*
objects in a volume *V*, the cube root of *V*/*N*, the typical distance between
neighbours in a uniform distribution. For a million objects in the unit cube,
*s* = 0.01.

**On uniform data, start between 1.5 and 2.5 mean separations**, a compromise
between the optima of the different queries. If one kind of query dominates,
move towards its optimum:

- **nearest and *N* nearest:** lower, 1 to 1.5 mean separations;
- **balls and shells:** higher, since the work grows with the radius measured
  in cells. A cell side of about 1.5 times the radius is a good choice, and
  the cell side should never equal the radius: a ball whose radius reaches
  the cell side makes each query visit a further layer of cells, and was 1.2
  to 1.7 times slower than at the best side.

**On clustered data, measure.** The optimum depends on how strongly the data
cluster and where the queries fall, and on the clustered set of the
performance page it differs from the uniform one. Build grids at a few cell
sides between about 0.5 and 4 mean separations and time a batch of queries
like your own. `benchmarks/benchmark_cellsize.py` does this for the sets of
the performance page and reports the occupancy of the cells; its source shows
the steps to repeat on your own points.

The measurements are on the [performance page](benchmark.md#cell-side).

## Queries

There are four queries, each of which can be asked about a point or about an
object already in the grid.

- `nearestObject` returns the index of the nearest object.
- `nearestObjects(N, …)` returns the *N* nearest, nearest first. *N* may be 0,
  which returns nothing, and may not exceed the number of objects that can be
  results.
- `closeObjects(…, Rmax, Rmin)` returns every object at a distance *d* with
  `Rmin <= d <= Rmax`: a *ball* when `Rmin` is 0, its default, and a *shell*
  otherwise. Both ends are included, so `Rmin == Rmax` selects the objects at
  exactly that distance. `Rmax` may be `+inf`. A ball extending past the box
  is truncated at its faces.
- `countObjects` returns the number of objects `closeObjects` would return
  for the same arguments, without collecting them.

```{literalinclude} ../examples/example.py
:language: python
:start-at: "nearest = grid.nearest_object(0.5, 0.5, 0.5)"
:end-at: "around = grid.nearest_objects(5, nearest)"
```

```{literalinclude} ../examples/example.cpp
:language: cpp
:start-at: "const unsigned int nearest = grid.nearestObject(0.5, 0.5, 0.5);"
:end-at: "const std::vector<unsigned int> around = grid.nearestObjects(5, nearest);"
```

**A query about an object excludes that object by identity.** Another object
at the same position is a different object, at distance 0, and is returned.
A query about a point excludes nothing: an object lying on the point is at
distance 0 and is returned.

**Distances** are Euclidean, with the [minimum image](#periodic-boxes) along
periodic axes. They are compared through their squares, computed in double
precision, so an object exactly on a radius is decided by that comparison
alone. Objects at equal distance are ranked by ascending index, and a tie at
the *N*-th place keeps the lower indices. The assignment of objects to cells
never changes a result. The order of the indices `closeObjects` returns
carries no meaning; see [reproducibility](#reproducibility).

## Testing points and indices

`contains(x, y, z)` reports whether a point lies in the box, which is the
test every member taking a point applies. It is false for a NaN coordinate,
and never raises. `isAlive(index)` (`is_alive` in Python) reports whether an
index names an object still in the grid; it is the one member taking an index
that does not raise, and returns false for an index out of range or removed,
and in Python for a negative one. Both also take a batch and return one flag
per entry: a `std::vector<std::uint8_t>` of 0 and 1 in C++, a `bool` array in
Python. They serve to screen a batch before it is queried.

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "accepted = grid.contains(bad_x, qy, qz)"
:end-at: "assert alive.tolist() == [True, True, True, False, False]"
```

```{literalinclude} ../examples/example_batch.cpp
:language: cpp
:start-at: "std::vector<double> screened = QX;"
:end-at: "const std::vector<std::uint8_t> alive = grid.isAlive(asked);"
```

Three members expose the cells: `get_cell` gives the linear index of the cell
holding a point, `get_cellCoords` (`get_cell_coords`) its coordinates
`(ix, iy, iz)`, and `get_objectsInCell` (`get_objects_in_cell`) the objects in
it. The linear index is `(ix * ny + iy) * nz + iz`, with `(nx, ny, nz)` the
cells per axis from `get_nCells()` (`n_cells`). They serve to inspect a grid;
no query requires them.

## Batches

Every query also exists as a *batch*: one call that answers many query points,
or many objects already in the grid. A batch has the name of the single
query, and is selected by its argument: three vectors (C++) or three arrays
(Python) in place of three numbers, or a vector or array of indices in place
of one index.

```{literalinclude} ../examples/example_batch.cpp
:language: cpp
:start-at: "const std::vector<unsigned int> nearest = grid.nearestObject(QX, QY, QZ);"
:end-at: "printf(\"query 12 has %llu neighbours within %.2f\\n\", (unsigned long long)(last-first), radius);"
```

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "nearest = grid.nearest_object(qx, qy, qz)"
:end-at: "five = grid.nearest_objects(5, qx, qy, qz)"
```

**Results are in input order.** The result of query `i` is in position `i`,
and equals that of the single query, in the same order, although a large
batch is processed internally in the order of the cells.

**A batch is faster than a loop of single queries**, in Python by about the
cost of a call per query; the [performance page](benchmark.md#per-call) gives
the figures.

**The shape of the result.** The batches of `nearestObject` and
`nearestObjects` are rectangular: a vector of
`n*N` in C++, with query `i`'s at `[i*N, (i+1)*N)`, and an array of shape
`(n_query, N)` in Python. `countObjects` returns one count per query, as a
`std::vector<std::uint64_t>` or an `int64` array.

A batch of `closeObjects` returns two arrays, since each query finds a
different number of objects: `indices`, every index found,
concatenated in query order, and `offsets`, with one more entry than there
are queries, starting at 0 and ending at the length of `indices`. The result
of query `i` is `indices[offsets[i]]` up to, excluding, `indices[offsets[i+1]]`;
a query that found nothing has two equal offsets. In C++ the two are the
members of a `MeshGrid::Batch`; in Python, `close_objects` returns a named
tuple `meshsearch.Batch` with the fields `indices` and `offsets`, which also
unpacks as a pair. The offsets are 64-bit (`std::uint64_t`; `int64` in
Python), since a batch can find more than 2³² objects in total.

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "indices, offsets = grid.close_objects(qx, qy, qz, radius)"
:end-at: "assert np.array_equal(counts, within)"
```

With the two flat arrays, per-query quantities are single vectorised
expressions: pair each result with its query, then reduce with `np.bincount`
or select with a mask.

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "query_of = np.repeat(np.arange(len(qx)), counts)"
:end-at: "mean_distance = np.divide(summed, counts, out=np.zeros(len(qx)), where=counts > 0)"
```

**Batches over objects** take their indices, and exclude each object from its
own result.

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "objects = nearest[:1000]"
:end-at: "assert np.array_equal(np.diff(ring_offsets), around)"
```

```{literalinclude} ../examples/example_batch.cpp
:language: cpp
:start-at: "const std::vector<unsigned int> objects = {0, 1, 2};"
:end-at: "const std::vector<std::uint64_t> around = grid.countObjects(objects, radius);"
```

**Arguments in Python.** Query points may be any one-dimensional array-like of
real numbers: a float64 or integer array is used as it is, and anything else,
such as a list or a float32 array, is converted by numpy first. Indices may
be any one-dimensional array-like of integers; an integer array of any dtype
is used as it is, never cast, so a negative index, or one not in the grid,
raises `meshsearch.IndexError`. A list or tuple is converted by numpy first.
An argument that has a length makes the call a batch, so a one-element list
returns a one-element result.

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "queries = rng.random((50_000, 3))"
:end-at: "qz = np.ascontiguousarray(queries[:, 2])"
```

## Reusing memory between calls

**In C++**, each query that returns a container also has a form that writes
into one passed by the caller as the last argument: a
`std::vector<unsigned int>` for indices, a `std::vector<std::uint64_t>` for
counts, a `std::vector<std::uint8_t>` for flags, a `MeshGrid::Batch` for a
batch of balls or shells. The container is cleared, then filled, and keeps
its capacity, so with single queries and small batches a container declared
outside a loop stops allocating once it has reached the largest result. A
large batch still allocates working space on every call, and the cell-ordered
batch of `closeObjects` allocates its `indices` anew. In the index batches of `nearestObject` and
`nearestObjects` the container may be the index vector itself. If the call
throws, the content of the container is unspecified.

```{literalinclude} ../examples/example_batch.cpp
:language: cpp
:start-at: "std::vector<unsigned int> found;"
:end-at: "if (rings.indices.size() != totals.back()) return 1;"
```

**In Python**, the batches of `nearest_object` (`uint32`, shape
`(n_query,)`), `nearest_objects` (`uint32`, `(n_query, n)`), `count_objects`
(`int64`, `(n_query,)`), `contains` and `is_alive` (`bool`, `(n_query,)`)
accept an array to fill through the keyword-only argument `out=`, and return
it. The array must have exactly that dtype and shape, and be C-contiguous and
writeable; a C-contiguous view into a larger array, such as one row of a
table, is accepted. Nothing is converted: a wrong dtype, a read-only array or
an object that is not an array raises `TypeError`, and a wrong shape or
layout raises `meshsearch.Error`. The array is checked before any query runs,
and is not written to if the call raises. `close_objects` has no `out=`;
`count_objects` gives the totals in advance.

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "neighbours = np.empty((len(qx), 5), dtype=np.uint32)"
:end-at: "assert np.array_equal(neighbours, five)"
```

## Periodic boxes

Any axis of the grid can be *periodic*, as in a simulation box: distances
along it are measured the short way round, as if the box repeated without end
along that axis. This is the *minimum image* convention: of all the periodic
images of an object, a query considers only the nearest, and so finds each
object at most once.

```{literalinclude} ../examples/example_periodic.py
:language: python
:start-at: "cellsize = 2 * (1.0 / n) ** (1 / 3)"
:end-at: "limits=[[0, 1], [0, 1], [0, 1]], periodic=True)"
```

```{literalinclude} ../examples/example_periodic.cpp
:language: cpp
:start-at: "const double cellsize = 2.*std::cbrt(1./double(n));"
:end-at: "meshsearch::MeshGrid grid(X, Y, Z, cellsize, options);"
```

In Python, `periodic` is one bool for all three axes or a triple, one per
axis, and `grid.periodic` returns the flags; in C++ they are
`Options::periodic` and `get_periodic()`. Periodic and non-periodic axes can
be mixed, for instance in a slab periodic in two directions.

**The limits are the box.** A periodic axis requires limits, and its period is
`hi - lo`. Its box is exactly the limits, neither widened nor extended to
whole cells. Coordinates are never wrapped: a point outside the limits is an
error, so coordinates must be wrapped into the box before the grid is built.
Points at `lo` and at `hi` are both accepted, and along that axis their
distance is 0. An axis of zero extent cannot be periodic.

**The periods must share one cell side.** All cells are cubes of one side, so
every period must be an integer multiple of the shortest: a box of 1 by 1 by 2
is accepted, 1 by 1 by 1.5 is not. The match is checked to a relative 10⁻¹²,
which absorbs the rounding of limits such as `0.1 * 3`.

**The cell side used can be smaller than the one requested.** The grid uses
the largest side that divides the shortest period without exceeding the
requested one: with a period of 1 and a request of 0.234, five cells of 0.2.
`get_cellsize()` (`cellsize` in Python) returns the side used. Non-periodic
axes use the same side and are extended to whole cells as usual. With no
periodic axis, the side is exactly the one requested.

**Any radius is valid.** A ball may be larger than the box, and `Rmax` may be
`+inf`: since each object is found once, at its nearest image, a ball of
infinite radius returns every object exactly once, and `nearestObjects` can
request as many objects as the grid holds. Along a periodic axis the search
covers at most half a period on either side of the query point.

```{literalinclude} ../examples/example_periodic.py
:language: python
:start-at: "nearest = grid.nearest_object(left)"
:end-at: "corner = grid.close_objects(0.0, 0.0, 0.0, 0.1)"
```

```{literalinclude} ../examples/example_periodic.cpp
:language: cpp
:start-at: "const unsigned int nearest = grid.nearestObject(left);"
:end-at: "const std::vector<unsigned int> corner = grid.closeObjects(0., 0., 0., 0.1);"
```

**The cost.** A periodic query also visits the cells across the faces, and
tests the wrap in the end cells of each periodic axis. Batched queries on a
grid periodic along all three axes took up to 50 per cent longer than on the
same points without periodicity; see the
[performance page](benchmark.md#periodic-boxes). A grid with no periodic axis
does none of this work.

## Changing the grid

`addObject(x, y, z)` adds an object and returns its index, one past the
highest ever issued; the point must lie in the box, which is never refitted.
`removeObject(index)` removes an object, whose index then remains invalid for
the life of the grid.

```{literalinclude} ../examples/example.py
:language: python
:start-at: "grid.remove_object(nearest)"
:end-at: "added = grid.add_object(0.5, 0.5, 0.5)"
```

Neither call releases memory: a removed object's storage remains allocated,
and a cell that fills up moves its objects to a larger block, leaving the old
one unused. Additions and removals suit occasional changes. **To add more
than a few per cent of the objects, build a new grid** from the full set of
coordinates; the new grid is compact.

## Memory

What a grid allocates is determined by the number of objects and of cells.
Each object costs **37 bytes**: its coordinates and the bookkeeping that maps
indices to storage. Each cell costs **at least 112 bytes** on a grid with no
periodic axis: 16 for the cell and 96 for a table of cell offsets the queries
walk, with eight entries per cell. Along a periodic axis the table needs half
as many entries, so a grid periodic along all three axes spends about 28 bytes
per cell. At one mean separation there are about as many cells as objects. At ten million objects and one mean
separation, this is 353 MiB for the objects and at least 1,076 MiB for the
cells, about 1.4 GiB; at two mean separations there are an eighth as many
cells. Empty cells cost the same as full ones, which matters on clustered
data.

Construction needs more for a while: a temporary cell index of 8 bytes per
object, and in Python a copy of the three coordinate arrays (24 bytes per
object), 76 and 229 MiB at ten million objects. Both are freed when the
constructor returns, but the allocator may retain the memory in the process.
The growth of the process therefore varies between runs; the
[performance page](benchmark.md#memory) gives the measurements.

## Errors

Errors are exceptions, never return codes or empty results.

| raised | C++ | Python |
|---|---|---|
| on any invalid input or impossible query | `meshsearch::Error`, derived from `std::runtime_error` | `meshsearch.Error`, derived from `ValueError` |
| on an index out of range or removed | `meshsearch::IndexError`, derived from `Error` | `meshsearch.IndexError`, derived from both `meshsearch.Error` and the built-in `IndexError` |
| when memory runs out | `std::bad_alloc` | `MemoryError` |
| on an argument of the wrong type | (a compile error) | `TypeError` |

Catching `Error` therefore catches every error the grid itself raises, index
errors included, but not exhaustion of memory or, in Python, a `TypeError`.
`Error` is raised for:

- the constructor: coordinate arrays of different lengths, a coordinate that
  is not finite or lies outside the limits, a cell side that is not finite and
  positive, limits that are not finite or have `lo > hi`, no points and no
  limits, a periodic axis without limits or whose period is not an integer
  multiple of the shortest or is below 2⁻¹⁰⁰⁰, and a grid too large to address
  (a cell side tiny against the box, more than 4,294,967,295 objects, or a box
  so large that distances across it overflow a double). Python also requires
  the limits to be three pairs. A failed constructor builds nothing.
- a query point outside the box, and any point query on a default-constructed
  C++ grid, which has no box;
- a negative or NaN radius, or `Rmin > Rmax`;
- an *N* larger than the number of objects that can be results, a negative
  *N* in Python, an empty grid for the nearest object, and `nearestObject`
  about an object when no other object remains (its batch: fewer than two
  objects, even for no index);
- a batch of `nearestObjects` whose result, queries times *N*, is too large
  to count;
- a batch whose coordinate arrays differ in length, or, in Python, an
  argument that is not one-dimensional;
- `addObject` for a point outside the box, or once 4,294,967,295 indices have
  been issued.

`IndexError` is raised by every member taking an index, except `isAlive`, for
an index out of range or removed, and in Python for a negative index, which is
never counted from the end. In Python a float where an index is expected
raises `TypeError`.

**A batch raises wherever one of its queries would, and returns no partial
result.** In Python the message names the first query that failed, as in
`query 99: X coordinate outside the box`. The checks that do not depend on
the query points (the radii, *N*, the array lengths, an empty grid) are made
even for an empty batch. An `addObject` that fails leaves the grid unchanged.

## Threads

The queries, and every other `const` member in C++, may be called
concurrently on one grid from any number of threads, and each call returns
what it would return alone. `addObject` and `removeObject` may not run while
any other thread uses the grid; the grid takes no lock, so the caller must
ensure this. A container passed to a C++ query, or an `out=` array in Python,
must not be accessed by another thread until the call returns. The library
starts no threads: a batch runs on the calling thread, and parallel work is
divided by the caller, for instance by giving each thread a slice of the
query points.

In Python, a batch releases the GIL while its queries run, so other Python
threads continue and may query the same grid. The module declares that it
does not require the GIL, so on a free-threaded build of CPython (3.14t)
importing it leaves the GIL disabled, and threads query in parallel under the
same rules.

## Reproducibility

**The same input gives the same results, in the same order, on Linux and
macOS, on x86-64 and ARM, and with any number of threads.** The same input
means the same coordinates in the same order, the same cell side and options,
the same sequence of additions and removals, and the same queries. This
covers the order of the indices `closeObjects` returns, the order of objects
within a cell that `get_objectsInCell` reports, and every batch; a batch's
result equals that of its single queries. A copy of a grid answers every
query as the original does, in the same order.

The order of the objects within a cell, and therefore the order of the
results of `closeObjects` and `get_objectsInCell`, is not part of the
interface: it may change between versions, and any such change is announced
in the [changelog](https://github.com/SimoSartori/meshsearch/blob/main/CHANGELOG.md).
Code that requires a particular order should sort the results.

## Copies and the version

A grid is copied by its copy constructor and assignment in C++, and by
`copy.copy` and `copy.deepcopy` in Python. A copy is deep and independent:
changing one grid leaves the other unchanged. Moves in C++ transfer the
storage, and leave the source fit only to be destroyed or assigned to. A default-constructed C++ grid holds nothing: every query raises
or returns empty until a grid is assigned to it.

Grids cannot be saved to a file, and a Python grid cannot be pickled. To use
a grid in several processes, rebuild it in each from the coordinates, or use
threads, which share one grid.

`meshsearch.__version__` in Python, and `MESHSEARCH_VERSION` in C++ from the
header `meshsearch/Version.h`, give the version the library was built as; the
header also defines `MESHSEARCH_VERSION_MAJOR`, `_MINOR` and `_PATCH`.
