# Guide

What is worth knowing before the reference pages. Everything here applies to
both APIs unless it says otherwise.

## Indices are stable, and errors are exceptions

```{include} ../README.md
:parser: myst_parser.sphinx_
:start-after: <!-- docs:concepts-start -->
:end-before: <!-- docs:concepts-end -->
```

## The cell side

The cell side is the one tuning parameter. Too large and every query tests the
many objects sharing a cell; too small and a query walks many cells to cover the
same sphere. One to four mean separations of the data is the usual range, and
two to three is a reasonable default for mixed use.

What each choice costs is measured, at two data sizes and for all three query
kinds, on the [performance page](benchmark.md). For a shell query the number
that decides the work is `Rmax / cellsize` rather than either alone: keep it at
or below 1.

The constructor takes the box as well, and if it is not given, fits it to the
data and grows each side to a whole number of cells — so `get_lims` (`lims` in
Python) can report a box slightly wider than the points, and a point just
outside the data range may still be accepted. The grown box always contains the
box it was given: where rounding would leave it a hair inside, one more cell is
added, so a point exactly on a limit is always accepted.

Distances are compared through their squares, computed in double precision as
`dx*dx + dy*dy + dz*dz`. `closeObjects` returns every object whose squared
distance lies between `Rmin*Rmin` and `Rmax*Rmax`, both included; the
nearest-neighbour queries rank objects by the same quantity, ties broken by
index. The result never depends on how objects fall into cells.

A point within rounding of a face may be
assigned to the cell on either side of it, and every member agrees on which,
but no result depends on that choice. For this to hold, every distance inside
the box has to be a finite double, so a box whose squared diagonal overflows —
an extent of about 1e154 along an axis — is refused, as is a cellsize whose
reciprocal overflows.

An axis along which the box has no extent — every point at the same coordinate,
or limits with `lo == hi` — gets one cell, and the box is grown to one cell side
around it. Flat and needle-shaped point sets are therefore accepted. A box that
cannot be counted is refused with an error instead: more than 2147483647 cells
along one axis, which a cell side tiny against the box would ask for.

## Periodic boxes

A simulation box is usually periodic: a particle leaving through one face comes
back through the opposite one, and two objects near opposite faces are close
neighbours. Each axis can be made periodic on its own, and distances along it
are then measured the short way round.

```{literalinclude} ../examples/example_periodic.cpp
:language: cpp
:start-at: "const double cellsize = 4./std::cbrt(double(n));"
:end-at: "grid.get_nCells()[0], grid.get_nCells()[1], grid.get_nCells()[2]);"
```

In Python the same grid is built with the keywords `limits` and `periodic`,
neither of which can be given by position. `periodic` is one bool for all three
axes or a triple, and `grid.periodic` reads it back:

```{literalinclude} ../examples/example_periodic.py
:language: python
:start-at: "cellsize = 4 * n ** (-1 / 3)"
:end-at: "f\"used {grid.cellsize:.6f}, {'x'.join(str(c) for c in grid.n_cells)} cells\")"
```

**The box is the limits.** A periodic axis requires `limits`, because the box
cannot be inferred from the data: the period is `L = hi - lo`, and the box is
exactly the limits, neither padded nor grown to whole cells, so `get_lims`
returns what was given. Coordinates are not wrapped. A point outside the limits
is an error, as everywhere else, and one at `lo` and one at `hi` are two objects
at opposite faces, a period apart.

**The minimum image.** Along a periodic axis the displacement from the query to
an object is `dx = y - q`, replaced by `dx - L` if `dx >= L/2` and by `dx + L`
if `dx < -L/2`. Every displacement then lies in `[-L/2, L/2)`, which is the
nearest of the object's periodic images, and the other images are never
considered, so an object is returned at most once whatever the radius. A
displacement of exactly `+L/2` is half a period either way round; it is taken as
`-L/2`, which changes nothing in the distance. Both corrections are exact in
floating point: they add no rounding to `dx` beyond that of the subtraction
itself. The squared distance is then formed as always, from the three
displacements.

**Any radius is valid.** A ball or shell may be larger than the box, and
`+inf` is accepted: `closeObjects` with an infinite `Rmax` returns every object
once. `nearestObjects` can ask for any `N` up to the number of objects, since
each object has exactly one nearest image, and it returns that many.

```{literalinclude} ../examples/example_periodic.cpp
:language: cpp
:start-at: "const unsigned int nearest = grid.nearestObject(left);"
:end-at: "printf(\"objects within 0.1 of the corner (0, 0, 0): %zu\\n\", corner.size());"
```

```{literalinclude} ../examples/example_periodic.py
:language: python
:start-at: "nearest = grid.nearest_object(left)"
:end-at: "print(f\"objects within 0.1 of the corner (0, 0, 0): {corner.size}\")"
```

**The sides must match.** A periodic box has a single cell side, so the periodic
sides have to be whole multiples of the shortest one: a box 1 by 1 by 2 is
fine, 1 by 1 by 1.5 is not. The agreement is checked to a relative 1e-12, which
absorbs the rounding of limits such as `0.1 * 3`, and anything further apart is
an `Error`. The axes need not otherwise be the same, and periodic and
non-periodic axes can be mixed; the non-periodic ones use the same cell side
and grow to whole cells as described above. An axis with no extent cannot be
periodic.

**The effective cell side.** The cell side has to divide the shortest period
exactly, so the grid uses the largest side that does not exceed the one
requested: the shortest period is split into the smallest whole number of cells
no larger than `cellsize`, and every periodic axis holds a whole number of them.
With a period of 1 and a requested side of 0.234 the grid uses 0.2, five cells
per period. The side actually used is never larger than the one requested, and is
usually only slightly smaller; read it with `get_cellsize()` (`cellsize` in
Python) rather than assuming it, for example when choosing a radius relative to
the cell. When no axis is periodic it is the requested side, exactly.

**The cost.** A periodic query also visits the cells on the far side of the
boundary, which is work a non-periodic query on the same data does not do, plus
a wrap test in the cells at the ends of the box. Expect a periodic grid to
answer queries between a few percent and about 20 percent slower than a
non-periodic one over the same points and cells, the larger figure with all
three axes periodic. A grid with no periodic axis runs the same code, at the same
speed, as before the option existed, and returns the same results.

## Batched queries

Every query also comes as a batch, in both languages: over many points, or
over many objects already in the grid. A batch has the name of the single
query; what tells them apart is the argument: three vectors (C++) or three
arrays (Python) in place of three numbers, or a vector or array of indices in
place of one index. In C++:

```{literalinclude} ../examples/example_batch.cpp
:language: cpp
:start-at: "const std::vector<unsigned int> nearest = grid.nearestObject(QX, QY, QZ);"
:end-at: "printf(\"query 12 has %llu neighbours within %.2f\\n\", (unsigned long long)(last-first), radius);"
```

In Python the query points can be any array-like. What selects the batch is an
argument that has a length, in place of a number, so a one-element list or
array is still a batch and returns a one-element result. A float64 or integer
array is read as it is, without a copy, whatever its strides; anything else,
such as a list, a tuple or a float32 array, is converted by numpy first, and
must hold real numbers (`TypeError` otherwise) in one dimension
(`meshsearch.Error` otherwise):

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "queries = rng.random((50_000, 3))"
:end-at: "qz = np.ascontiguousarray(queries[:, 2])"
```

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "nearest = grid.nearest_object(qx, qy, qz)"
:end-at: "five = grid.nearest_objects(5, qx, qy, qz)"
```

**Results are in input order.** Query `i`'s answer is in position `i`, exactly
the answer the single query gives for that point or object, in the same order within it.
How the batch is worked through internally is not visible in the result: a
large batch, of a hundred thousand queries or more, is answered in an order
that follows the grid's cells, so that consecutive queries read neighbouring
memory, and is then put back into input order. That is faster than the same
queries asked one by one, and in Python it also saves the few microseconds that
crossing into the module costs per call, which can be more than a query itself
takes.

### Batches over objects

A batch over objects takes their indices, and each object is excluded from its
own answer, as in the single index query. In C++ the indices are a
`std::vector<unsigned int>`:

```{literalinclude} ../examples/example_batch.cpp
:language: cpp
:start-at: "const std::vector<unsigned int> objects = {0, 1, 2};"
:end-at: "(unsigned long long)around[0], (unsigned long long)around[1], (unsigned long long)around[2]);"
```

In Python they are a one-dimensional array-like of integers: the `uint32`
arrays the queries return, numpy's default `int64`, or a list. An integer
array of any dtype is read as it is, never converted, so an index cannot wrap
round to another object: a negative value, one too large to be an index, or a
removed object raises `meshsearch.IndexError`, naming the query. Anything that
is not integers raises `TypeError`. The same holds for a single index: `-1` is
not the last object but an index out of range, and raises
`meshsearch.IndexError`. A negative `n` raises `meshsearch.Error`.

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "objects = nearest[:1000]"
:end-at: "assert np.array_equal(np.diff(ring_offsets), around)"
```

### Errors

A batch raises wherever one of its queries would, and returns
nothing partial. Python names the first query that fails, as in
`query 99: X coordinate outside the box`; C++ raises the single query's
`Error` or `IndexError`. The checks that do not depend on a query — on the
radii, on `N` against the number of objects, on the coordinate arrays having
the same length — are made even for an empty batch.

### The shape of a batch result

`nearestObject` returns one index per query, and `nearestObjects` returns
exactly `N` per query, so their batches are rectangular: a vector of `n*N` with
query `i`'s at `[i*N, (i+1)*N)` in C++, an `(n_query, N)` array in Python.

A shell is not rectangular, since the number found varies per query.
`closeObjects` therefore returns the compressed-row form: one array of every
index found, concatenated in query order, and an array of offsets with one
more entry than there are queries, starting at 0 and ending at the number of
indices. The results of query `i` run from `offsets[i]` up to, excluding,
`offsets[i+1]`; a query that found nothing has two equal offsets, never a
missing entry. In C++ the two arrays are the members `indices` and `offsets` of
a `MeshGrid::Batch`; in Python `close_objects` returns them as a named tuple,
`meshsearch.Batch`, with the fields `indices` and `offsets`, which also unpacks
like a plain pair. The
offsets are 64-bit, because a batch can find more results in total than 32 bits
can count: `std::uint64_t` in C++, and `int64` in Python, numpy's standard
integer, which indexes, slices and repeats without a cast.

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "indices, offsets = grid.close_objects(qx, qy, qz, radius)"
:end-at: "assert np.array_equal(counts, within)"
```

What makes the form worth learning is pairing each result with the query it
came from. `indices` and `query_of` below are two flat arrays of the same
length, so any per-query quantity is one vectorised expression over every
result at once: `np.bincount(query_of, weights=...)` reduces per query, and
boolean masks select across queries. A list of arrays would cost one Python
object per query, which is the overhead the batch exists to remove, so none is
offered:

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "query_of = np.repeat(np.arange(len(qx)), counts)"
:end-at: "mean_distance = np.divide(summed, counts, out=np.zeros(len(qx)), where=counts > 0)"
```

## Counting neighbours

`countObjects` (`count_objects` in Python) returns how many objects
`closeObjects` would return, with the same arguments and the same closed
interval `Rmin <= d <= Rmax`, without collecting them. When only the number
matters, for a density estimate or a pair count, it is the cheaper call. It has
the same four forms as `closeObjects`: a point or an object index, one query or
a batch. A batch of counts is a vector of `std::uint64_t` in C++ and an `int64`
array in Python, in input order; a single count is a `std::size_t` or a Python
`int`.

```{literalinclude} ../examples/example_batch.cpp
:language: cpp
:start-at: "const std::vector<std::uint64_t> within = grid.countObjects(QX, QY, QZ, radius);"
:end-at: "printf(\"query 12 counted: %llu\\n\", (unsigned long long)within[12]);"
```

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "radius = 0.02"
:end-at: "within = grid.count_objects(qx, qy, qz, radius)"
```

## Testing points and indices

`contains(X, Y, Z)` says whether a query at the point would be accepted, that
is, whether the point lies in the box: the very test every member taking a point
applies, so a point for which it is false is one on which they raise `Error`.
It is false for a NaN coordinate, and never raises. `isAlive(index)` (`is_alive`
in Python) is the one member taking an index that does not raise: an index that
is out of range or removed answers false, a negative one included in Python.
Both have a batch, returning one flag per entry: a `std::vector<std::uint8_t>`
holding 0 or 1 in C++, a `bool` array in Python, and both have the container
form in C++ and `out=` in Python. Use them to screen a batch before asking it,
instead of catching the error it raises.

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "accepted = grid.contains(bad_x, qy, qz)"
:end-at: "assert alive.tolist() == [True, True, True, False, False]"
```

```{literalinclude} ../examples/example_batch.cpp
:language: cpp
:start-at: "std::vector<double> screened = QX;"
:end-at: "accepted[99], alive[0], alive[1], alive[2]);"
```

## Copies

A `MeshGrid` is copied by its copy constructor and assignment, which are deep:
the copy shares no storage with the original apart from a table that depends
only on the geometry and is never modified, and answers every query as the
original does, the order of the objects within a cell included. In Python
`copy.copy(grid)` and `copy.deepcopy(grid)` do the same. Moves transfer the
storage without copying it, and leave the source grid fit only to be destroyed
or assigned to.

Grids are not serializable, and a Python `MeshGrid` cannot be pickled. For
parallel work use threads, which share one grid, or rebuild the grid in each
process from the coordinates, which is fast.

## The version

`meshsearch.__version__` in Python and `MESHSEARCH_VERSION` in
`meshsearch/Version.h`, with `MESHSEARCH_VERSION_MAJOR`, `_MINOR` and `_PATCH`,
are the version the library was built as. Both come from the project version in
`CMakeLists.txt`, the one place it is written for them.

## Reusing a container in C++

Each C++ query that returns a container also has a form that writes into one
the caller passes, always as the last argument: a `std::vector<unsigned int>`
for indices, a `std::vector<std::uint64_t>` for counts, a `MeshGrid::Batch` for
a batch of shells. The container is cleared and then filled, and it keeps the
capacity it had. In a loop, declare it once outside the loop: after the first
few calls it has grown to the largest answer seen, and from then on no call
allocates.

```{literalinclude} ../examples/example_batch.cpp
:language: cpp
:start-at: "std::vector<unsigned int> found;"
:end-at: "(unsigned long long)totals[0], (unsigned long long)totals[1], (unsigned long long)totals[2]);"
```

The `closeObjects` and `countObjects` forms come with and without `Rmin`, as
the returning forms do through the default argument, so the container is last
either way. In the two index batches of `nearestObject` and `nearestObjects`,
the container may be the vector of indices itself, which is then replaced by
the answers. If the call throws, the container's content is unspecified.

## Reusing an array in Python

Python returns new arrays, but some batches have a result whose shape is known
before they run, and accept an array to write into instead, through the
keyword-only argument `out=`: the batched `nearest_object`, whose result is
`(n_query,)`, the batched `nearest_objects`, whose result is `(n_query, n)`,
and the batched `count_objects`, whose result is `(n_query,)`, over points or
over objects alike, and the batched `contains` and `is_alive`, whose result is
`(n_query,)`. The call fills the array and returns it.

```{literalinclude} ../examples/example_batch.py
:language: python
:start-at: "neighbours = np.empty((len(qx), 5), dtype=np.uint32)"
:end-at: "assert np.array_equal(neighbours, five)"
```

The array must already be exactly the result: `uint32` for the indices of
`nearest_object` and `nearest_objects`, `int64` for counts and `bool` for
`contains` and `is_alive`, of the exact shape, C-contiguous and
writeable. Anything else is an error, never a silent conversion, since a
converted copy would be a different array and the one passed in would stay
unwritten. A wrong dtype, a read-only array, or something that is not an array
raises `TypeError`; a wrong shape or a non-contiguous layout raises
`meshsearch.Error`. A C-contiguous view into a larger array, such as one row of
a table, is accepted, and only its elements are written. The array is checked
before any query runs and is not written to if the call raises.

`close_objects` has no `out=`: how many indices a batch of shells finds is
known only once it has run, so no array could be supplied in advance. Use
`count_objects` first if the number is needed beforehand.

## Threads

Const members — every query — may be called concurrently on one grid from any
number of threads; `addObject` and `removeObject` may not run alongside
anything else on the same grid. The library starts no threads of its own: a
batch runs on the thread that calls it.

In Python, a batched call releases the GIL while the queries run, so other
Python threads keep running meanwhile, and several threads can query one grid
at the same time. The caller is then responsible for the rule above: do not
modify a grid with `add_object` or `remove_object` while other threads query
it. A single-point query holds the GIL throughout.
