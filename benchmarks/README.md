# Benchmarks

Two scripts measure meshsearch from Python. They produced the figures on the
documentation's [performance page](https://simosartori.github.io/meshsearch/benchmark.html)
(`docs/benchmark.md`). They are not part of the package: nothing installs
them, and neither the library nor its tests use them.

- `benchmark.py` compares meshsearch with scipy's `cKDTree` and
  scikit-learn's `BallTree`: construction time, memory, and the nearest
  object, the ten nearest and a ball of two mean separations, per call and
  batched.
- `benchmark_cellsize.py` sweeps meshsearch's cell side from one to four mean
  separations and reports, for each, construction time, memory, how the
  objects fill the cells, and the same three queries.

The *mean separation* of *N* points in a volume *V* is the cube root of
*V*/*N*. Cell sides and radii are given in mean separations.

## Inputs and method

Both scripts generate their own points in the unit cube, from a fixed seed:
`uniform`, or `clustered`, a Thomas process with nine tenths of the points in
clusters of about 1000 (the docstring of `benchmark.py` gives the
parameters). Query points come from the same distribution. `benchmark.py`
draws 10,000 new query points for each of three runs, the same draws for every
library, and reports the median, so no run finds what it visits already in
cache from the one before; `--queries repeat` runs the same 1000 points three
times instead. `benchmark_cellsize.py` runs the same 1000 points three times.
Each row runs in a process of its own, so
that the memory figure, the growth of the process's resident size during the
build, belongs to one structure. Every meshsearch grid is built with the unit
cube as its limits.

## Running them

Build meshsearch in Release mode, from a source checkout, and install the
other libraries:

```
pip install -C cmake.build-type=Release .
pip install scipy scikit-learn psutil
```

(`pip install ".[benchmark]"` installs the same dependencies, with meshsearch
built in the default mode.) Then:

```
python benchmarks/benchmark.py                                      # uniform and clustered, cell sides 1 and 4
python benchmarks/benchmark.py --dists clustered --factors 4        # a selection
python benchmarks/benchmark.py --periodic --sizes 1000000 10000000  # all axes periodic, against cKDTree(boxsize=1)
python benchmarks/benchmark.py --large-batch --sizes 1000000 10000000
python benchmarks/benchmark_cellsize.py --sweep
python benchmarks/benchmark_cellsize.py --sweep --dists clustered
```

| option | script | meaning |
|---|---|---|
| `--sizes N …` | `benchmark.py` | numbers of points; default 10⁵, 10⁶ and 10⁷ |
| `--factors F …` | `benchmark.py` | meshsearch cell sides, in mean separations; default 1 and 4 |
| `--dists D …` | both | `uniform`, `clustered` or both (default) |
| `--periodic` | `benchmark.py` | all three axes periodic; `BallTree` is left out |
| `--large-batch` | `benchmark.py` | one call of 10⁵ and of 10⁶ uniform query points, at each of `--factors` |
| `--reps R` | `benchmark.py` | repetitions of each large batch (default 3) |
| `--queries fresh\|repeat` | `benchmark.py` | new query points for every run (default), or the same ones |
| `--sweep` | `benchmark_cellsize.py` | every size (10⁶, 10⁷), cell side and input |
| `--one LIBRARY N`, `--one-sweep N F` | each | a single row, as JSON; what the scripts run in each subprocess |

Each script prints its rows as JSON on standard output and its progress on
standard error. On an Apple M1 Pro the comparison, periodic and large-batch
runs took 17 minutes together, most of it in `BallTree`'s per-call radius
queries at ten million points.

The scripts measure the two generated inputs only. To choose a cell side for
your own points, repeat what `benchmark_cellsize.py` does on them: build grids
at a few cell sides and time a batch of queries like your own.
