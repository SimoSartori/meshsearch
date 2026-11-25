# Performance

The cost of meshsearch's construction, memory and queries, measured from
Python against scipy's `cKDTree` and scikit-learn's `BallTree`, and the effect
of the cell side. The advice drawn from these measurements is in the
[guide](guide.md#choosing-the-cell-side). Every measured number, at every size,
is in the [data files](#data-files).

## Summary

Ten million points, a cell side of one mean separation *s*, batched queries
of 10,000 fresh points in random order. Times of queries are in microseconds
(µs) per query; *periodic* is the uniform set with all three axes periodic,
against `cKDTree` with `boxsize=1`.

| | uniform: meshsearch | cKDTree | clustered: meshsearch | cKDTree | periodic: meshsearch | cKDTree |
|---|---|---|---|---|---|---|
| construction (s) | 2.67 | 6.23 | 2.37 | 5.99 | 1.82 | 6.15 |
| memory (MiB) | 1,573 | 399 | 1,054 | 335 | 969 | 428 |
| nearest (µs per query) | 1.14 | 2.89 | 2.04 | 2.94 | 1.54 | 3.50 |
| k = 10 (µs per query) | 5.24 | 6.12 | 3.45 | 5.32 | 6.16 | 8.12 |
| radius 2 *s* (µs per query) | 9.51 | 9.82 | 12.66 | 44.70 | 11.85 | 15.93 |

Construction is 2.3 to 3.4 times faster than `cKDTree`'s. The nearest object
is found 1.4 to 2.5 times faster and the ten nearest 1.2 to 1.5 times faster;
the radius query is level on uniform data, 1.3 times faster on the periodic
set and 3.5 times faster on clustered data. The cost is memory: the process
grows by 2.3 to 3.9 times as much as with `cKDTree`.

## Method

| | |
|---|---|
| machine | Apple M1 Pro, 10 cores, 16 GiB, macOS 14.5 |
| compiler | Apple clang 15, CMake Release build (`-O3`) |
| Python | 3.14.7, numpy 2.5.5, scipy 1.18.1, scikit-learn 1.9.1 |

Each row is one library at one size, run in its own process on one thread,
with the machine otherwise idle. A query time is the median of three runs.
Each run draws 10,000 new query points in random order, the same draws for
every library, so that no run finds the cells or tree nodes it visits already
in the processor's cache. *Batched* is one call for all 10,000 points; *per
call* is one Python call per point. *Memory* is the growth of the process's
resident size during construction; it includes temporary memory that the
allocator retained. Every structure is built with its default parameters,
meshsearch with the unit cube as explicit limits. Differences below about
10 per cent are within the variation between runs.

## Data

The mean separation *s* of *N* points in a volume *V* is the cube root of *V*/*N*. Cell
sides and radii are given in units of *s*; the radius query uses 2 *s* and the
k-nearest query k = 10.

- **Uniform:** points and query points uniform in the unit cube. A ball of
  radius 2 *s* holds about 33 objects.
- **Clustered:** a Thomas process in the unit cube. Nine tenths of the points
  lie in clusters of about 1,000, each a Gaussian of width *s* around a
  uniformly drawn centre; the rest are uniform. Query points are fresh draws
  from the same clusters, so a ball of radius 2 *s* holds about 410 objects.

## Against cKDTree and BallTree

Batched queries. The trees have no cell side; meshsearch is shown at 1 *s* and
4 *s*. Sizes of 10⁵ points are in the data files.

### Uniform

| points | | meshsearch, 1 *s* | meshsearch, 4 *s* | cKDTree | BallTree |
|---|---|---|---|---|---|
| **10⁶** | construction (s) | 0.190 | 0.135 | 0.367 | 0.369 |
| | memory (MiB) | 266 | 92 | 52 | 96 |
| | nearest (µs per query) | 0.95 | 2.62 | 1.55 | 17.52 |
| | k = 10 (µs per query) | 3.35 | 3.53 | 3.57 | 28.90 |
| | radius (µs per query) | 7.18 | 1.70 | 7.36 | 9.08 |
| **10⁷** | construction (s) | 2.670 | 1.480 | 6.225 | 6.838 |
| | memory (MiB) | 1,573 | 683 | 399 | 176 |
| | nearest (µs per query) | 1.14 | 3.21 | 2.89 | 63.76 |
| | k = 10 (µs per query) | 5.24 | 4.97 | 6.12 | 89.53 |
| | radius (µs per query) | 9.51 | 2.35 | 9.82 | 18.06 |

### Clustered

| points | | meshsearch, 1 *s* | meshsearch, 4 *s* | cKDTree | BallTree |
|---|---|---|---|---|---|
| **10⁶** | construction (s) | 0.183 | 0.131 | 0.359 | 0.362 |
| | memory (MiB) | 194 | 85 | 49 | 95 |
| | nearest (µs per query) | 1.25 | 4.65 | 1.77 | 47.00 |
| | k = 10 (µs per query) | 3.02 | 4.01 | 3.92 | 66.23 |
| | radius (µs per query) | 9.03 | 5.40 | 36.80 | 24.57 |
| **10⁷** | construction (s) | 2.372 | 1.393 | 5.986 | 6.593 |
| | memory (MiB) | 1,054 | 584 | 335 | 178 |
| | nearest (µs per query) | 2.04 | 5.72 | 2.94 | 145.98 |
| | k = 10 (µs per query) | 3.45 | 5.99 | 5.32 | 199.38 |
| | radius (µs per query) | 12.66 | 6.73 | 44.70 | 44.10 |

Construction is 1.9 to 4.7 times faster than either tree's. At 1 *s*
meshsearch is faster than or level with `cKDTree` on every query. At 4 *s*
its radius query is 4.2 to 6.8 times faster than `cKDTree`'s, while the
nearest object is 1.1 to 2.6 times slower. `BallTree` is the slowest of the
three on every query except the clustered radius query, where it is faster
than `cKDTree`.

## Periodic boxes

All three axes periodic, against `cKDTree` with `boxsize=1`; `BallTree` has
no periodic mode. Batched queries. The cell side must divide the period, so
at 10⁷ points the grid used 0.004630 for a requested 0.004642 (1 *s*) and
0.018519 for 0.018566 (4 *s*).

| data | points | | meshsearch, 1 *s* | meshsearch, 4 *s* | cKDTree, boxsize |
|---|---|---|---|---|---|
| uniform | **10⁶** | construction (s) | 0.169 | 0.134 | 0.414 |
| | | memory (MiB) | 106 | 77 | 50 |
| | | nearest (µs per query) | 1.32 | 2.55 | 2.53 |
| | | k = 10 (µs per query) | 5.03 | 3.78 | 5.85 |
| | | radius (µs per query) | 9.33 | 1.88 | 11.02 |
| | **10⁷** | construction (s) | 1.815 | 1.406 | 6.149 |
| | | memory (MiB) | 969 | 686 | 428 |
| | | nearest (µs per query) | 1.54 | 3.31 | 3.50 |
| | | k = 10 (µs per query) | 6.16 | 4.87 | 8.12 |
| | | radius (µs per query) | 11.85 | 2.61 | 15.93 |
| clustered | **10⁶** | construction (s) | 0.154 | 0.135 | 0.367 |
| | | memory (MiB) | 91 | 77 | 51 |
| | | nearest (µs per query) | 1.53 | 4.91 | 2.46 |
| | | k = 10 (µs per query) | 3.46 | 4.22 | 5.58 |
| | | radius (µs per query) | 11.09 | 5.89 | 46.43 |
| | **10⁷** | construction (s) | 1.878 | 1.381 | 6.117 |
| | | memory (MiB) | 864 | 552 | 363 |
| | | nearest (µs per query) | 2.37 | 5.91 | 3.42 |
| | | k = 10 (µs per query) | 4.61 | 6.05 | 7.99 |
| | | radius (µs per query) | 16.23 | 6.80 | 58.37 |

Against the same grid without periodic axes, batched queries took between
3 per cent less and 50 per cent more time. At 1 *s* meshsearch remains faster
than `cKDTree` on every query.

## Per call

One Python call per query point, in µs per query, with the same fresh query
points as the batched tables.

| data | points | query | meshsearch, 1 *s* | meshsearch, 4 *s* | cKDTree | BallTree |
|---|---|---|---|---|---|---|
| uniform | 10⁶ | nearest | 1.69 | 3.22 | 9.30 | 49.29 |
| | | k = 10 | 5.16 | 4.85 | 13.02 | 60.83 |
| | | radius | 8.96 | 3.16 | 9.31 | 750.29 |
| | 10⁷ | nearest | 2.28 | 3.81 | 9.88 | 97.06 |
| | | k = 10 | 7.11 | 6.38 | 14.22 | 123.80 |
| | | radius | 11.89 | 4.08 | 12.69 | 6,881.01 |
| clustered | 10⁶ | nearest | 2.01 | 5.11 | 9.01 | 80.33 |
| | | k = 10 | 4.30 | 5.34 | 12.08 | 101.68 |
| | | radius | 10.80 | 6.92 | 29.72 | 758.65 |
| | 10⁷ | nearest | 2.78 | 6.36 | 10.08 | 187.03 |
| | | k = 10 | 5.26 | 6.87 | 13.93 | 237.07 |
| | | radius | 14.51 | 8.18 | 37.43 | 6,854.37 |

A meshsearch call costs 0.5 to 1.1 µs more than the same query in a batch for
the nearest object, and up to 2.4 µs more for the radius query; a `cKDTree`
call for the nearest object costs 9 to 10 µs. meshsearch is faster per call on
every query in every row; the periodic rows are in the data files.

The cost of the call itself, measured separately on 10⁶ uniform points at
1 *s* by repeating a few query points (median of five runs of 20,000 calls):

| query | µs per call |
|---|---|
| nearest, three Python floats | 1.06 |
| nearest, a numpy float64 row unpacked | 1.49 |
| k = 10 | 4.38 |
| radius 2 *s* | 8.01 |
| `contains` | 0.05 |
| `is_alive` | 0.04 |

With scipy and scikit-learn imported into the same process, the figures change
by less than 10 per cent.

## Large batches

A batch of 10⁵ query points or more is answered internally in the order of
the cells, and the results are returned in input order. One call of 10⁵ or
10⁶ fresh uniform query points in random order; the *input order* columns are
the same points asked in batches of 50,000, which are answered in the order
given. In µs per query.

| grid | queries | query | meshsearch, 1 *s* | input order, 1 *s* | meshsearch, 4 *s* | input order, 4 *s* | cKDTree |
|---|---|---|---|---|---|---|---|
| 10⁶ | 10⁵ | nearest | 0.44 | 1.02 | 2.15 | 2.54 | 1.81 |
| 10⁶ | 10⁵ | k = 10 | 1.89 | 3.95 | 2.92 | 3.60 | 4.03 |
| 10⁶ | 10⁵ | radius | 3.88 | 7.50 | 1.21 | 1.60 | 7.60 |
| 10⁶ | 10⁶ | nearest | 0.53 | 0.98 | 2.25 | 2.52 | 1.79 |
| 10⁶ | 10⁶ | k = 10 | 1.89 | 3.95 | 3.00 | 3.60 | 3.95 |
| 10⁶ | 10⁶ | radius | 3.77 | 7.38 | 1.34 | 1.69 | 7.49 |
| 10⁷ | 10⁵ | nearest | 0.89 | 1.42 | 2.45 | 3.48 | 2.46 |
| 10⁷ | 10⁵ | k = 10 | 2.93 | 5.68 | 3.32 | 5.27 | 5.26 |
| 10⁷ | 10⁵ | radius | 5.07 | 10.46 | 1.53 | 2.31 | 9.83 |
| 10⁷ | 10⁶ | nearest | 0.58 | 1.40 | 2.43 | 3.19 | 2.51 |
| 10⁷ | 10⁶ | k = 10 | 2.08 | 5.48 | 3.22 | 4.80 | 5.28 |
| 10⁷ | 10⁶ | radius | 4.24 | 10.57 | 1.40 | 2.33 | 9.90 |

The cell order makes queries 1.6 to 2.6 times faster at 1 *s* and 1.1 to 1.7
times faster at 4 *s*. A program that can present its query points in spatial
order obtains the same gain with batches of any size. On large batches at
1 *s* meshsearch is 1.8 to 4.3 times faster than `cKDTree` on every query.

## Cell side

Batched query times at ten million points and six cell sides, with the
occupancy of the cells: the fraction of empty cells, the mean count in the
non-empty ones and the 99th percentile of the count over all cells. The radius
stays 2 *s*. Query points are drawn as for the other tables; the sweep was run
five times and each time is the median of the five. The sweep at 10⁶ points
is in the data files.

### Uniform

| cell side | cells | empty | mean in non-empty | 99th percentile | construction (s) | memory (MiB) | nearest (µs per query) | k = 10 (µs per query) | radius (µs per query) |
|---|---|---|---|---|---|---|---|---|---|
| 1 *s* | 10,077,696 | 37 % | 1.6 | 4 | 2.24 | 2,210 | 1.21 | 4.91 | 9.20 |
| 1.5 *s* | 2,985,984 | 3.5 % | 3.5 | 8 | 1.82 | 1,133 | 1.32 | 3.17 | 4.43 |
| 2 *s* | 1,259,712 | 0 % | 7.9 | 15 | 1.66 | 849 | 1.52 | 3.34 | 3.95 |
| 2.5 *s* | 658,503 | 0 % | 15.2 | 25 | 1.59 | 769 | 1.77 | 3.62 | 2.59 |
| 3 *s* | 373,248 | 0 % | 26.8 | 40 | 1.55 | 733 | 2.06 | 3.95 | 2.39 |
| 4 *s* | 157,464 | 0 % | 63.5 | 83 | 1.46 | 696 | 3.15 | 4.66 | 2.27 |

### Clustered

| cell side | cells | empty | mean in non-empty | 99th percentile | construction (s) | memory (MiB) | nearest (µs per query) | k = 10 (µs per query) | radius (µs per query) |
|---|---|---|---|---|---|---|---|---|---|
| 1 *s* | 10,077,696 | 80 % | 5.0 | 24 | 2.17 | 2,068 | 1.68 | 3.50 | 10.86 |
| 1.5 *s* | 2,985,984 | 59 % | 8.2 | 76 | 1.72 | 1,035 | 2.26 | 3.87 | 7.86 |
| 2 *s* | 1,259,712 | 35 % | 12.2 | 165 | 1.58 | 736 | 2.85 | 4.24 | 7.67 |
| 2.5 *s* | 658,503 | 16 % | 18.1 | 284 | 1.51 | 651 | 3.39 | 4.38 | 5.94 |
| 3 *s* | 373,248 | 4.4 % | 28.0 | 434 | 1.45 | 605 | 3.97 | 4.58 | 5.92 |
| 4 *s* | 157,464 | 0.1 % | 63.6 | 730 | 1.40 | 580 | 5.39 | 5.22 | 6.18 |

On uniform data the nearest object is fastest at 1 *s*, with 1.5 *s* 9 per
cent slower, and 2.6 times slower at 4 *s*. The ten nearest are fastest at
1.5 *s*, with 2 *s* 5 per cent slower. The radius query is fastest at 4 *s*,
with 3 *s* 5 per cent and 2.5 *s* 14 per cent slower, and four times slower
at 1 *s*. At 10⁶ points the optima are the same within one step of the sweep.

On clustered data the nearest and the ten nearest are fastest at 1 *s*, where
four fifths of the cells are empty. They slow down as the cells grow: at 4 *s*
one cell in a hundred holds 730 objects or more, eleven times the mean, and
the nearest object is 3.2 times slower than at 1 *s*. The radius query is
fastest at 2.5 to 3 *s*.

**The step at a cell side equal to the radius.** At 2 *s* the cell side
equals the radius, and a ball whose radius is at least the cell side costs
more than one just below it. A separate measurement on 10⁶ uniform points
(median of five runs of 10,000 fresh queries, batched):

| cell side | 1.98 *s* | 1.999 *s* | 2.000 *s* | 2.001 *s* | 2.02 *s* | 2.98 *s* | 3.00 *s* | 3.02 *s* |
|---|---|---|---|---|---|---|---|---|
| radius / cell side | 1.010 | 1.0005 | 1.000 | 0.9995 | 0.990 | 0.671 | 0.667 | 0.662 |
| radius (µs per query) | 2.28 | 2.34 | 2.27 | 1.80 | 1.55 | 1.44 | 1.50 | 1.55 |

In the sweep the step appears as the gain from 2 *s* to 2.5 *s*: the radius
query is 1.5 times faster on uniform and 1.3 times faster on clustered data.

## Memory

What a grid allocates is determined by the number of objects and of cells:
37 bytes per object (24 for the coordinates, 13 for the bookkeeping that maps
indices to storage) and, with no periodic axis, at least 112 bytes per cell
(16 for the cell and 96 for its eight entries in a table of cell offsets); a
grid periodic along all three axes needs about 28 bytes per cell. At 10⁷
points and 1 *s*, with 10,077,696 cells, this is 353 MiB for the objects and
at least 1,076 MiB for the cells, about 1.4 GiB. At 4 *s* the 157,464 cells
take about 17 MiB in addition to the same 353 MiB.

Construction needs more while it runs: a temporary cell index of 8 bytes per
object (76 MiB at 10⁷) and, from Python, a copy of the three coordinate
arrays, 24 bytes per object (229 MiB). Both are freed when the grid is built,
but the allocator may retain them in the process. This is why the tables
report about 700 MiB for the grid of 10⁷ points at 4 *s*; a C++ program
building the same grid grew by 455 MiB. The grid is built compact, with no
capacity reserved for later additions.

The growth of the resident size is not the allocation, and it varies between
runs. At 10⁷ points and 4 *s* it varied by less than 3 per cent. At 1 *s*,
identical runs grew by 1.0 to 2.2 GiB: in some runs less
than the 1.4 GiB the grid allocates, in others considerably more. The cause of
the growth below the allocation has not been determined.

## Not measured

Threads; additions and removals; boxes elongated along one axis; clustering of
other strengths or shapes; radii other than 2 *s*; the batched `count_objects`
and `contains` against the trees; machines other than the one above.

## Data files

Every number measured for this page, at every size, in
[`benchmarks/results/`](https://github.com/SimoSartori/meshsearch/tree/main/benchmarks/results).
Times are in µs per query, construction in seconds, memory in MiB.

- [`comparison.csv`](https://github.com/SimoSartori/meshsearch/blob/main/benchmarks/results/comparison.csv):
  meshsearch at 1 *s* and 4 *s*, `cKDTree` and `BallTree`, uniform and
  clustered, 10⁵, 10⁶ and 10⁷ points, per call and batched.
- [`periodic.csv`](https://github.com/SimoSartori/meshsearch/blob/main/benchmarks/results/periodic.csv):
  the periodic boxes, per call and batched.
- [`large_batches.csv`](https://github.com/SimoSartori/meshsearch/blob/main/benchmarks/results/large_batches.csv):
  the large batches, in cell order and in input order.
- [`cellside.csv`](https://github.com/SimoSartori/meshsearch/blob/main/benchmarks/results/cellside.csv)
  and [`cellside_runs.csv`](https://github.com/SimoSartori/meshsearch/blob/main/benchmarks/results/cellside_runs.csv):
  the cell-side sweep at 10⁶ and 10⁷ points, medians and each of the five
  runs, batched and per call.
- [`radius_step.csv`](https://github.com/SimoSartori/meshsearch/blob/main/benchmarks/results/radius_step.csv):
  the radius query around a cell side equal to the radius.
- [`call_overhead.csv`](https://github.com/SimoSartori/meshsearch/blob/main/benchmarks/results/call_overhead.csv):
  the cost of one Python call.
- [`memory_cpp.csv`](https://github.com/SimoSartori/meshsearch/blob/main/benchmarks/results/memory_cpp.csv):
  the construction measured from C++.

The scripts are in `benchmarks/`; `benchmarks/README.md` describes each and
its options. From a source checkout:

```
pip install -C cmake.build-type=Release .
pip install scipy scikit-learn psutil
python benchmarks/benchmark.py
python benchmarks/benchmark.py --periodic --sizes 1000000 10000000
python benchmarks/benchmark.py --large-batch --sizes 1000000 10000000
python benchmarks/benchmark_cellsize.py --sweep
```

Each prints its rows as JSON. The first three took 17 minutes together on the
machine above, most of it in `BallTree`'s per-call radius queries at 10⁷
points.
