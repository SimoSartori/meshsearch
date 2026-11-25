# meshsearch

meshsearch is a C++17 library, with Python bindings, for exact neighbour
searches in three-dimensional point sets. It is faster than the tree
structures usually employed for the task: on ten million points it builds its
index two to three and a half times faster than scipy's `cKDTree`, matches or
outperforms it on batched queries with a suitable cell side, and runs fixed-radius searches on clustered
data several times faster.

It achieves this with a cell list, a uniform mesh of cubic cells over the
domain, so that each query examines only the cells that can hold a result.
Results are reproducible: the same input gives the same results, in the same
order, on Linux and macOS, on x86-64 and ARM. Any axis of the domain may be
periodic.

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

The cell side, 0.2 here, is two *mean separations* of these points, the
typical distance between neighbours. The [guide](guide.md#choosing-the-cell-side)
explains how to choose it, and why the limits are given explicitly.

## The pages

- [Guide](guide.md): the concepts and their use, in the order a new user
  needs them, with starting values.
- [C++ reference](cpp.rst) and [Python reference](python.rst): every member,
  its arguments and the errors it raises.
- [Names side by side](naming.md): the C++ and Python spelling of each member.
- [Performance](benchmark.md): the cost of construction, memory and queries,
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
