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

## Batched queries, and the form they return

```{include} ../README.md
:parser: myst_parser.sphinx_
:start-after: <!-- docs:batched-start -->
:end-before: <!-- docs:batched-end -->
```
