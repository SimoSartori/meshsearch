# Changelog

## Unreleased

### Added

- C++: periodic axes. `MeshGrid::Options` holds the `limits` and a per-axis
  `periodic` flag, and `get_periodic()` returns the flags. On a periodic axis
  distances are minimum-image distances and any radius is valid.
- Python: `MeshGrid(x, y, z, cellsize, *, limits=None, periodic=False)`.
  `periodic` is a bool for all three axes or a triple of bools. The read-only
  property `periodic` returns a tuple of three bools. `cellsize` can be
  slightly below the value requested when an axis is periodic.
- Examples `examples/example_periodic.cpp` and `examples/example_periodic.py`,
  and a section on periodic boxes in the guide.
- C++: `countObjects`, the number of objects `closeObjects` would return,
  for a point or an object index.
- C++: batches of every query, under the same names, taking
  `const std::vector<double>&` X, Y, Z or `const std::vector<unsigned int>&`
  indices: `nearestObject` returns one index per query, `nearestObjects` `n*N`
  indices, `closeObjects` a `MeshGrid::Batch` (`indices` and 64-bit `offsets`),
  `countObjects` a `std::vector<std::uint64_t>`. Results are in input order; a
  large batch is answered internally in cell order, which is faster.
- C++: every query returning a container has an overload writing into one
  passed as the last argument, cleared and then filled, its capacity kept.
- Python: `count_objects(x, y, z, rmax, rmin=0)` and
  `count_objects(index, rmax, rmin=0)`, and their batches over arrays of points
  or of object indices, returning `int64` counts.
- Python: batches over arrays of object indices for `nearest_object`,
  `nearest_objects` and `close_objects`. The indices may be of any integer
  dtype and are not converted: a negative or too large value raises
  `meshsearch.IndexError`.
- Python: keyword-only `out=` on the batched `nearest_objects` and
  `count_objects`: a C-contiguous array of exactly the result's dtype and shape,
  filled and returned. Any other array is an error, never converted.
- Examples `examples/example_batch.cpp` and additions to
  `examples/example_batch.py`, and sections of the guide on batches, counts,
  container forms, `out=` and threads.

### Changed (Python interface)

- **Breaking:** `limits` is keyword-only. `MeshGrid(x, y, z, cellsize, limits)`
  raises `TypeError`; write `MeshGrid(x, y, z, cellsize, limits=limits)`.
- **Breaking:** the offsets returned by the batched `close_objects` are
  `int64`, formerly `uint32`.
- The batched queries run through the C++ batches, so a large batch is
  answered in cell order; results are unchanged and still in query order. The
  GIL is released while they run, as before.
- A batch whose arguments the grid refuses regardless of the points — an `n`
  above the number of objects for `nearest_objects`, invalid radii for
  `close_objects`, an empty grid for `nearest_object` — now raises even when
  the batch is empty.

### Changed (C++ interface)

- The constructor takes `const MeshGrid::Options&` in place of the limits
  `std::vector<std::vector<double>>`, which is removed. `MeshGrid(X, Y, Z,
  cellsize)` is unchanged. With no periodic axis the behaviour and every
  result are as before.

### Fixed

- `nearestObject` (both overloads; `nearest_object` in Python) could return an
  object that was not the nearest when an object lay within a few ulp of a
  cell face: the search stopped one shell early with no allowance for
  rounding.
- `closeObjects` (both overloads; `close_objects` in Python) could miss an
  object at exactly `Rmax` when it lay within a few ulp of a cell face, and
  could miss objects at or beyond `Rmin` when the cellsize was comparable to
  the spacing of doubles at the coordinates' magnitude (|coordinate| /
  cellsize around 1e15 or more). Both ends of the range of shells it visits
  now carry a proven margin for rounding.
- The constructor could grow the box to slightly *inside* the given limits,
  when the range divided by the cellsize rounded down onto a whole number,
  and then refuse a point lying exactly on a limit. The grown box now always
  contains the limits: where rounding would leave it inside, one more cell is
  added along that axis. For limits where that happens, `get_nCells` and
  `get_lims` report one more cell than 0.2.1, and the order of `closeObjects`
  results can differ from 0.2.1.
- Distances above about 1.3e154 overflowed to infinity, so `nearestObject`
  could report that no object was found while objects were present, and
  `closeObjects` could return objects beyond `Rmax`. Such boxes are now
  refused (see below).
- A cellsize below about 5.6e-309 made cell assignment undefined behaviour.
  It is now refused (see below).
- Python: a batched `nearest_objects` call allocated `n_query * n` entries
  before checking `n`, so an `n` larger than the grid could fail with
  `MemoryError`, and a product beyond 64 bits could wrap around. The array is
  now sized after the first query has succeeded, so such an `n` raises that
  query's `meshsearch.Error`, and a product that 64-bit sizes cannot count
  raises `meshsearch.Error`.

### Changed

- The constructor raises `meshsearch::Error` for a cellsize whose reciprocal
  overflows a double, and for a box whose squared diagonal, computed from the
  grown box, overflows a double (an extent of about 1e154 along an axis).
  Within an accepted box every distance is a finite double.
- A search no longer stops early while the distance it has covered is below
  about 1e-150, where squared distances are subnormal and rounding errors are
  no longer relative; nor while the threshold it compares against is above
  about 1e301. With a cellsize that small a query may visit every cell.
  Results are unchanged.
- `closeObjects` with `Rmin > 0` starts its walk up to two shells closer to
  the query than before. Results and their order are unchanged.

## 0.2.1 — 2026-10-20

### Changed

- `nearestObjects` (both overloads; `nearest_objects` in Python) now orders
  objects at equal distance by ascending index, and a tie at the N-th place
  keeps the lower indices. The order of ties used to come from the standard
  library's heap, so it differed between libc++ and libstdc++. Results for
  inputs with exactly tied distances change on every compiler; results
  without ties are unchanged.

## 0.2.0 — 2026-10-07

### Fixed

- Grids with more than 46,340 cells along one axis, or more generally with
  nx² + ny² + nz² above 2³¹ − 1, are now built and searched correctly. The
  cell-offset arithmetic used to overflow 32-bit integers there, which was
  undefined behaviour.
- A cellsize tiny against the box, an infinite limit or an infinite coordinate
  no longer leads to undefined behaviour while the cells are counted. Each now
  raises `meshsearch::Error` (see below).

### Changed

- The constructor now raises `meshsearch::Error`, before allocating anything,
  for:
  - a cellsize that is not finite (`+inf` used to be accepted);
  - limits that are not finite, or that have `lo > hi` on an axis;
  - a coordinate that is not finite;
  - more than 2,147,483,647 cells along an axis, or more cells or memory than
    64-bit sizes can count or address;
  - more than 4,294,967,295 objects.
- `addObject` raises `meshsearch::Error` once 4,294,967,295 indices have been
  issued.
- `std::bad_alloc` propagates untranslated, and is now documented
  (`MemoryError` in Python). An `addObject` that fails this way leaves the grid
  as it was.
- An axis of zero extent is documented: it gets one cell, with the box grown
  to one cellsize around it. This was already the behaviour.
- `GridOffset` and the protected members of `MeshGrid` are documented as
  implementation details, not part of the interface.
- **meshsearch is compiled with `-ffp-contract=off`.** Distances are no longer
  fused into FMA instructions where the compiler used to do so, which was on
  arm64 (Apple silicon, Linux aarch64). Results there may differ from 0.1.0
  at exact radius boundaries and between exactly equidistant objects. x86_64
  results are unchanged.

### Build and packaging

- CMake install rules and a package configuration:
  `find_package(meshsearch CONFIG)`, with the target `meshsearch::meshsearch`.
  That target name is also available when meshsearch is included with
  `add_subdirectory` or `FetchContent`. The `MESHSEARCH_INSTALL` option is on
  by default and off under scikit-build-core.
- As a subproject, meshsearch no longer builds its tests and example by
  default, and no longer sets `CMAKE_BUILD_TYPE`.
- The Python module no longer exports the library's C++ symbols.
- Python package metadata: SPDX licence expression (`BSD-3-Clause`), and
  Documentation, Source, Issues and Changelog URLs. Building from source needs
  scikit-build-core 0.11 or later.

## 0.1.0 — 2026-09-22

First release.
