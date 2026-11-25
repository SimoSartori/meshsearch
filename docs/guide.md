# Guide

What to know before the reference pages, in the order a new user meets it.
Everything applies to both languages unless it says otherwise; the
[names side by side](naming.md) give each member's spelling in the other one.

## Points, indices and the box

A grid is built from three coordinate arrays of equal length (vectors in C++,
any array-like in Python), a cell side, and optionally the limits of the box.
The coordinates are copied into the grid. Coordinates, cell side and radii are
in one unit, whichever it is; the library imposes none.

An *object* is a point held by the grid, and its *index* is its position in
the arrays passed to the constructor. Indices are stable for the life of the
grid: removing an object leaves its index permanently invalid, adding one
issues a new index, and no index is ever reused or renumbered. An index stored
anywhere in your program therefore never comes to mean a different object.

The *box* is the region in which the grid accepts points. Every member that
takes a point, a query point as much as a new object, raises `Error` for a
point outside it. There are two ways to set it.

- **Explicit limits**, `{{xlo, xhi}, {ylo, yhi}, {zlo, zhi}}`, are the region
  your points and every later query point belong to: the survey volume, the
  simulation box, the unit cube. Every coordinate passed to the constructor
  must lie within them.
- **Fitted limits**, when none are given, follow the points: the box is their
  range, widened by a tiny margin.

**Pass limits whenever a query point can fall outside the range of the
objects.** With fitted limits, a query point beyond the outermost object, or
a query drawn from a different sample than the objects, can raise `Error`
instead of returning its neighbours. Whether it does depends on how the box
was rounded to whole cells (below), which is not part of the interface, so do
not rely on the slack. Fitted limits are safe
only when every query point is itself one of the objects or known to lie
within their range. An empty set of points needs explicit limits.

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
periodic, it is widened symmetrically until the cells fit, so `get_lims()`
(`lims` in Python) can be slightly larger than the limits given, and always
contains them; a point in that extra margin is accepted. An axis along which
the box has no extent, because every point has the same coordinate or the
limits have `lo == hi`, gets a single cell, so flat and needle-shaped point
sets are accepted. On a periodic axis the box is exactly the limits; see
[periodic boxes](#periodic-boxes).

## Choosing the cell side

The cell side is the one tuning parameter. A query visits the cells that can
hold an answer and tests every object in them. With large cells it tests many
objects that are too far away; with small ones it walks many cells, most of
them empty, and the cells cost memory.

The scale to measure it against is the *mean separation* *s*: for *N* objects
in a volume *V*, the cube root of *V*/*N*, which is the typical distance
between neighbours if the objects were spread evenly. For a million objects in
the unit cube, *s* = 0.01.

**On uniform data, start at two mean separations.** The batched sweep on the
[performance page](benchmark.md#cell-side) measures, for uniform points:

- the nearest object is fastest at one mean separation, and becomes about
  three times slower at four with fresh query points (four to eight times
  when the queries' cells are already in cache);
- the ten nearest are fastest at two;
- a ball of radius two mean separations is fastest at three, and about five
  times slower at one;
- at two mean separations no query kind is more than about 1.7 times slower
  than at its own best side.

If one kind of query dominates the work, move towards its own best: one mean
separation for nearest-object queries, which also uses two to three times the
memory, or three to four for balls of about two mean separations. For a ball
or shell, the work grows with the radius measured in cells, so a radius much
larger than the cell side calls for a larger cell.

For radius queries, a cell side of 1 to 1.5 times the largest radius *R*max
was within 5 per cent of the best side on uniform data and on both clustered
sets measured: the one of the performance page, and one with clusters half as
wide. This was measured for one radius only, *R*max of two mean separations,
on a million points, batched, with 10,000 fresh query points per run; it is a
measured fact, not a rule for other radii.

**On clustered data the best side is different, and has been measured for one
kind of clustering only.** The performance page uses a set with nine tenths of
the points in compact clusters, where a ball of two mean separations holds
about twelve times as many objects as in uniform data. Measured against the
mean separation of the whole box:

- the nearest object and the ten nearest are fastest at one mean separation,
  where four fifths of the cells are empty;
- at four mean separations one cell in a hundred holds 730 or more objects,
  against an average of 64, and the nearest object is four to five times
  slower than at one;
- the ball of two mean separations is fastest at three.

How strongly your data cluster, and where the queries fall, both move these
optima, so for clustered data measure rather than guess: build grids at a few
cell sides, between about half and four mean separations, and time a batch of
queries like your own. `benchmarks/benchmark_cellsize.py` does exactly this
for the uniform and clustered sets of the performance page, and reports how
many objects each cell holds; its source shows the steps to repeat on your own
points. A cell side affects only speed and memory, never results.

## Queries

There are four queries, each of which can be asked about a point or about an
object already in the grid.

- `nearestObject` returns the index of the nearest object.
- `nearestObjects(N, …)` returns the *N* nearest, nearest first. *N* may be 0,
  which returns nothing, and may not exceed the number of objects that can be
  answers.
- `closeObjects(…, Rmax, Rmin)` returns every object at a distance *d* with
  `Rmin <= d <= Rmax`: a *ball* when `Rmin` is 0, its default, and a *shell*
  otherwise. Both ends are included, so an object at exactly either radius is
  returned, and `Rmin == Rmax` selects the objects at exactly that distance.
  `Rmax` may be `+inf`. A ball reaching past the box is cut off at its faces,
  not refused.
- `countObjects` returns how many objects `closeObjects` would return, with
  the same arguments, without collecting them: the cheaper call when only the
  number matters, as in a density estimate or a pair count.

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

**About an object, the object itself is excluded, by identity.** Asked about
object `i`, every query leaves `i` out of its own answer, but another object
at the same position is a different object, at distance 0, and is returned.
Asked about a point, nothing is excluded: an object lying on the point is at
distance 0 and is returned.

**Distances** are Euclidean, with the [minimum image](#periodic-boxes) along
periodic axes. They are compared through their squares, computed in double
precision, so an object exactly on a radius is decided by that comparison and
nothing else. Objects at equal distance are ranked by ascending index, and a
tie at the *N*-th place keeps the lower indices. How objects fall into cells
never changes a result.

`closeObjects` returns its indices in an order that carries no meaning; see
[reproducibility](#reproducibility) for what is promised about it.

## Testing points and indices

`contains(x, y, z)` tells whether a query at the point would be accepted, that
is, whether the point lies in the box: the very test every member taking a
point applies. It is false for a NaN coordinate, and never raises.
`isAlive(index)` (`is_alive` in Python) tells whether an index names an object
still in the grid; it is the one member taking an index that does not raise,
and answers false for an index out of range or removed, and in Python for a
negative one. Both also take a whole batch, and return one flag per entry: a
`std::vector<std::uint8_t>` of 0 and 1 in C++, a `bool` array in Python. Use
them to screen a batch before asking it, instead of catching the error it
raises.

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

Three members expose the cells themselves: `get_cell` gives the linear index
of the cell holding a point, `get_cellCoords` (`get_cell_coords`) its
coordinates `(ix, iy, iz)`, and `get_objectsInCell` (`get_objects_in_cell`)
the objects in it. The linear index is `(ix * ny + iy) * nz + iz`, with
`(nx, ny, nz)` the cells per axis from `get_nCells()` (`n_cells`). They are
for inspecting a grid, for instance to see how many objects the cells at your
query points hold; no query needs them.

## Batches

Every query also comes as a *batch*: one call that answers many query points,
or many objects already in the grid. A batch has the name of the single query;
what selects it is the argument: three vectors (C++) or three arrays (Python)
in place of three numbers, or a vector or array of indices in place of one
index.

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

**Results are in input order.** Query `i`'s answer is in position `i`, and is
exactly what the single query returns for that point or object, in the same
order. A large batch is worked through internally in an order that follows the
cells, so that consecutive queries read neighbouring memory, and its results
are then put back into input order; nothing of this is visible in the result.

**A batch is faster than a loop.** In Python a single call costs about as much
as a fast query itself: with fresh query points in random order, a
nearest-object query takes about 1.7 µs per call and about 1 µs per query in a
batch, at one mean separation and a million objects. A large batch also gains
from its cell order: 1.6 to 2.6 times at one mean separation, measured from
Python, down to about 0.5 µs for the nearest object. The [performance page](benchmark.md#per-call-or-batched) has the
figures.

**The shape of the result.** `nearestObject` returns one index per query and
`nearestObjects` exactly *N*, so their batches are rectangular: a vector of
`n*N` in C++, with query `i`'s at `[i*N, (i+1)*N)`, and an array of shape
`(n_query, N)` in Python. `countObjects` returns one count per query, as a
`std::vector<std::uint64_t>` or an `int64` array.

A batch of `closeObjects` finds a different number of objects for each query,
so it returns two arrays: `indices`, every index found, concatenated in query
order, and `offsets`, with one more entry than there are queries, starting at
0 and ending at the length of `indices`. The answer to query `i` is
`indices[offsets[i]]` up to, excluding, `indices[offsets[i+1]]`, and a query
that found nothing has two equal offsets. In C++ the two are the members of a
`MeshGrid::Batch`; in Python, `close_objects` returns a named tuple
`meshsearch.Batch` with the fields `indices` and `offsets`, which also unpacks
as a pair. The offsets are 64-bit (`std::uint64_t`, and `int64` in Python,
numpy's own index type), because a batch can find more than 2³² objects in
total.

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "indices, offsets = grid.close_objects(qx, qy, qz, radius)"
:end-at: "assert np.array_equal(counts, within)"
```

The two flat arrays make per-query quantities one vectorised expression: pair
each result with the query it came from, and reduce with `np.bincount` or
select with a mask. A list of arrays, one per query, would cost the Python
object per query that the batch exists to avoid, so none is offered.

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "query_of = np.repeat(np.arange(len(qx)), counts)"
:end-at: "mean_distance = np.divide(summed, counts, out=np.zeros(len(qx)), where=counts > 0)"
```

**Batches over objects** take their indices, and each object is excluded from
its own answer, as in the single query.

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

**What Python accepts.** Query points may be any one-dimensional array-like
of real numbers: a float64 or integer array is used as it is, and anything
else, such as a list or a float32 array, is converted by numpy first. Indices
may be any one-dimensional array-like of integers; an integer array of any
dtype is used as it is, never cast to another dtype, so a value can never wrap
round to another object: a negative index, or one that is not in the grid,
raises `meshsearch.IndexError`. A list or tuple is converted by numpy first. What
makes a call a batch is an argument that has a length, so a one-element list
is a batch and returns a one-element result.

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "queries = rng.random((50_000, 3))"
:end-at: "qz = np.ascontiguousarray(queries[:, 2])"
```

## Reusing memory between calls

**In C++**, each query that returns a container also has a form that writes
into one the caller passes as the last argument: a `std::vector<unsigned int>`
for indices, a `std::vector<std::uint64_t>` for counts, a `std::vector<std::uint8_t>`
for flags, a `MeshGrid::Batch` for a batch of balls or shells. The container is
cleared, then filled, and keeps its capacity, so with single queries and
small batches a container declared outside a loop stops allocating once it has
grown to the largest answer. A large batch still allocates working space on
every call, and the cell-ordered batch of `closeObjects` builds its `indices`
anew. The forms of
`closeObjects` and `countObjects` come with and without `Rmin`, so the
container is last either way. In the index batches of `nearestObject` and
`nearestObjects` the container may be the index vector itself. If the call
throws, the container's content is unspecified.

```{literalinclude} ../examples/example_batch.cpp
:language: cpp
:start-at: "std::vector<unsigned int> found;"
:end-at: "if (rings.indices.size() != totals.back()) return 1;"
```

**In Python**, a batch whose result has a shape known in advance accepts an
array to fill through the keyword-only argument `out=`, and returns it: the
batched `nearest_object` (`uint32`, shape `(n_query,)`), `nearest_objects`
(`uint32`, `(n_query, n)`), `count_objects` (`int64`, `(n_query,)`),
`contains` and `is_alive` (`bool`, `(n_query,)`). The array must already be
exactly the result: that dtype and shape, C-contiguous and writeable; a
C-contiguous view into a larger array, such as one row of a table, will do.
Nothing is converted, because a converted copy would be a different array and
yours would stay unwritten: a wrong dtype, a read-only array or something
that is not an array raises `TypeError`, and a wrong shape or layout raises
`meshsearch.Error`. The array is checked before any query runs, and is not
written to if the call raises.

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "neighbours = np.empty((len(qx), 5), dtype=np.uint32)"
:end-at: "assert np.array_equal(neighbours, five)"
```

`close_objects` has no `out=`: how many indices it finds is known only once it
has run. Call `count_objects` first if the total is needed beforehand.

## Periodic boxes

In a simulation box, an object leaving through one face comes back through
the opposite one, and two objects near opposite faces are close neighbours.
Any axis of the grid can be made *periodic*: distances along it are then
measured the short way round, as if the box repeated without end along that
axis. This is the *minimum image* convention: of all the periodic copies of an
object, a query sees only the nearest one, and so finds each object at most
once.

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
axis, and `grid.periodic` reads the flags back; in C++ they are
`Options::periodic` and `get_periodic()`. Periodic and non-periodic axes can
be mixed, for a slab periodic in two directions and bounded in the third.

**The limits are the box.** A periodic axis requires limits, and its period is
`hi - lo`. Its box is exactly the limits, neither widened nor grown to whole
cells, so `get_lims()` returns what was given. Coordinates are never wrapped:
a point outside the limits is an error, as on any other axis, so wrap your
coordinates into the box before building the grid. Points at `lo` and at `hi` are both
accepted, and along that axis their distance is 0.
An axis of zero extent cannot be periodic.

**The periods must fit one cell side.** All cells are cubes of one side, so
every periodic period must be a whole multiple of the shortest one: a box 1 by
1 by 2 is accepted, 1 by 1 by 1.5 is not. The match is checked to a relative
10⁻¹², which absorbs the rounding of limits such as `0.1 * 3`.

**The cell side used can be smaller than the one asked for.** The shortest
period has to hold a whole number of cells, so the grid uses the largest side
that does so without exceeding the requested one: with a period of 1 and a
request of 0.234, five cells of 0.2. Read the side actually used with
`get_cellsize()` (`cellsize` in Python) rather than assuming it. Non-periodic
axes use the same side and grow to whole cells as usual. With no periodic
axis, the side is exactly the one requested.

**Any radius is valid.** A ball may be larger than the box, and `Rmax` may be
`+inf`: since each object is seen once, at its nearest image, a ball of
infinite radius returns every object exactly once, and `nearestObjects` can
ask for as many objects as the grid holds. Along a periodic axis the search
covers at most half a period either side of the query point.

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

**Everything else is unchanged.** Batches, counts and the container forms work
on a periodic grid exactly as on any other, with the same distances.
`contains` applies the limits, and is false outside them, since points are
never wrapped. Ties, identity exclusion and the closed interval of
`closeObjects` are as described under [queries](#queries).

**The cost.** A periodic query also visits cells across the faces, and tests
the wrap in the cells at each end of a periodic axis. On the performance
page's measurements, batched queries on a grid periodic along all three axes
took up to about 45 per cent longer than on the same points without
periodicity, and some no longer at all.
At one mean separation they remained faster, for every query kind, than
scipy's `cKDTree` with its periodic `boxsize` option. A grid with no periodic axis does none of this work.

## Changing the grid

`addObject(x, y, z)` adds an object and returns its index, one past the
highest ever issued; the point must lie in the box, which is never refitted.
`removeObject(index)` removes one; its index then stays invalid for the life
of the grid, and every member taking it raises `IndexError`, except `isAlive`,
which answers false.

```{literalinclude} ../examples/example.py
:language: python
:start-at: "grid.remove_object(nearest)"
:end-at: "added = grid.add_object(0.5, 0.5, 0.5)"
```

Neither call returns memory: a removed object's storage stays allocated, and
a cell that fills up moves its objects to a larger block, leaving the old one
unused. Additions and removals suit occasional changes. **To add more than a
few per cent of the objects, build a new grid** from the full set of
coordinates: construction is fast, and the new grid is compact. Neither call
may run while any other thread uses the grid.

## Memory

A grid holds, for each object, its coordinates and bookkeeping: **37 bytes per
object**. Each cell costs **at least 112 bytes**: 16 for the cell itself and 96
for a table of cell offsets the queries walk, which has eight entries per cell
on a grid with no periodic axis. Along a periodic axis the table needs half as
many entries per cell, so a grid periodic along all three axes spends at least
28 bytes per cell. There are `nx * ny * nz` cells, so their share grows as the
cube of the box side over the cell side, and it dominates at fine cell sides:
at one mean separation there are about as many cells as objects, which adds
at least 112 bytes per object to the 37; at two, an eighth as many. On clustered data many cells
hold nothing, yet each still costs its share: with the box's own mean
separation as the cell side, four fifths of them were empty on the performance
page's clustered set.

Construction needs more for a while: a temporary cell index of 8 bytes per
object, and in Python a copy of the three coordinate arrays (24 bytes per
object) passed to the C++ constructor. Both are freed when the constructor
returns, but the allocator may keep the memory in the process. At ten million
objects and one mean separation the process grew by 1.1 to 2.1 GiB in repeated
measurements; the [performance page](benchmark.md#memory) has them.

A batch allocates its results, and a large batch some working space in
proportion to the number of queries. Removals and additions never shrink a
grid; see [changing the grid](#changing-the-grid).

## Errors

Errors are exceptions, never return codes or empty results.

| raised | C++ | Python |
|---|---|---|
| on any invalid input or impossible query | `meshsearch::Error`, derived from `std::runtime_error` | `meshsearch.Error`, derived from `ValueError` |
| on an index out of range or removed | `meshsearch::IndexError`, derived from `Error` | `meshsearch.IndexError`, derived from both `meshsearch.Error` and the built-in `IndexError` |
| when memory runs out | `std::bad_alloc` | `MemoryError` |
| on an argument of the wrong type | (a compile error) | `TypeError` |

Catching `Error` therefore catches every error the grid itself raises, index
errors included; it does not catch running out of memory or, in Python, a
`TypeError`. `Error` is raised for:

- the constructor: coordinate arrays of different lengths, a coordinate that
  is not finite or lies outside the limits, a cell side that is not finite and
  positive, limits that are not finite or have `lo > hi`, no points and no
  limits, a periodic axis without limits or whose period is not a whole
  multiple of the shortest or is below 2⁻¹⁰⁰⁰, and a grid too large to address
  (a cell side tiny against the box, more than 4,294,967,295 objects, or a box
  so large that distances across it overflow a double). Python also requires the limits to
  be three pairs. A failed constructor builds nothing.
- a query point outside the box, and any point query on a default-constructed
  C++ grid, which has no box;
- a negative or NaN radius, or `Rmin > Rmax`;
- an *N* larger than the number of objects that can be answers, a negative
  *N* in Python, an empty grid for the nearest object, and `nearestObject` about an object
  when no other object remains (its batch: fewer than two objects, even for no
  index);
- a batch of `nearestObjects` whose result, queries times *N*, is too large
  to count;
- a batch whose coordinate arrays differ in length, or, in Python, an
  argument that is not one-dimensional;
- `addObject` for a point outside the box, or once 4,294,967,295 indices have
  been issued.

In Python a negative index raises `meshsearch.IndexError`; it is never
counted from the end. A float where an index is expected raises `TypeError`.

**A batch raises wherever one of its queries would, and returns nothing
partial.** In Python the message names the first query that failed, as in
`query 99: X coordinate outside the box`. The checks that do not depend on the
query points (the radii, *N*, the array lengths, an empty grid) are made even
for an empty batch. An `addObject` that fails leaves the grid as it was.

## Threads

The queries, and every other `const` member in C++, may be called
concurrently on one grid from any number of threads, and each call returns
what it would return alone. `addObject` and `removeObject` may not run while
any other thread uses the grid; the grid takes no lock, so the caller must
ensure it. A container passed to a C++ query, or an `out=` array in Python,
must not be touched by another thread until the call returns. The library
starts no threads of its own: a batch runs on the thread that calls it, and
parallel work is the caller's to divide, for instance by giving each thread a
slice of the query points.

In Python, a batch releases the GIL while its queries run, so other Python
threads keep running and may query the same grid meanwhile. The module
declares that it does not need the GIL, so on a free-threaded build of CPython
(3.14t) importing it leaves the GIL disabled, and threads query truly in
parallel, under the same rules.

## Reproducibility

**The same input gives the same results, in the same order, on every platform
and with any number of threads.** The same input means the same coordinates in
the same order, the same cell side and options, the same sequence of
additions and removals, and the same queries. That covers the order of the
indices `closeObjects` returns, the order of objects within a cell that
`get_objectsInCell` reports, and every batch; a batch's result equals that of
its single queries. A copy of a grid answers every query as the original does,
in the same order.

Which order the objects take within a cell, and so the order of the results of
`closeObjects` and `get_objectsInCell`, is not part of the interface: it may
change between versions, and any such change is announced in the
[changelog](https://github.com/SimoSartori/meshsearch/blob/main/CHANGELOG.md).
Code that needs a particular order should sort the results.

## Copies and the version

A grid is copied by its copy constructor and assignment in C++, and by
`copy.copy` and `copy.deepcopy` in Python. A copy is deep and independent:
changing one grid leaves the other as it was. Moves in C++ transfer the
storage without copying it, and leave the source fit only to be destroyed or
assigned to. A default-constructed C++ grid holds nothing: every query raises
or returns empty until a grid is assigned to it.

Grids cannot be saved to a file, and a Python grid cannot be pickled. To use a
grid in several processes, rebuild it in each from the coordinates, which is
fast, or use threads, which share one grid.

`meshsearch.__version__` in Python, and `MESHSEARCH_VERSION` in C++ from the
header `meshsearch/Version.h`, give the version the library was built as; the
header also defines `MESHSEARCH_VERSION_MAJOR`, `_MINOR` and `_PATCH`.
