# meshsearch

meshsearch finds neighbours among points in three dimensions, for C++17 and
Python. It divides a box around the points into a uniform grid of cubic
*cells*, and answers each *query* by visiting only the cells near it: the
object nearest a point, the *N* nearest, every object within a radius or in a
spherical shell, and how many there are. A query can be asked about one point,
about an object already in the grid, or about a whole batch of either in one
call, and the box can be periodic along any axis.

The library prescribes no unit. Coordinates, the cell side and every radius
must simply be in the same one.

```{include} ../README.md
:parser: myst_parser.sphinx_
:start-after: <!-- docs:install-start -->
:end-before: <!-- docs:install-end -->
```

```{include} ../README.md
:parser: myst_parser.sphinx_
:start-after: <!-- docs:example-start -->
:end-before: <!-- docs:example-end -->
```

The cell side, 0.2 here, is two *mean separations* of these points: the
typical distance between neighbours. The [guide](guide.md#choosing-the-cell-side)
explains how to choose it, and why the limits are given explicitly.

## The pages

- [Guide](guide.md): what to know before the references, in the order a new
  user meets it, with starting values.
- [C++ reference](cpp.rst) and [Python reference](python.rst): every member,
  its arguments and the errors it raises.
- [Names side by side](naming.md): the C++ and Python spelling of each member.
- [Performance](benchmark.md): what queries, construction and memory cost,
  against scipy's `cKDTree` and scikit-learn's `BallTree`.

```{toctree}
:maxdepth: 2
:hidden:

guide
cpp
python
naming
benchmark
```
