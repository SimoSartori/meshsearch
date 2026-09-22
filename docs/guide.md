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
outside the data range may still be accepted.

## Batched queries, and the form they return

```{include} ../README.md
:parser: myst_parser.sphinx_
:start-after: <!-- docs:batched-start -->
:end-before: <!-- docs:batched-end -->
```
