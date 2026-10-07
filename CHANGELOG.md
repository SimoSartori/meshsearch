# Changelog

## 0.2.0 — unreleased

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
