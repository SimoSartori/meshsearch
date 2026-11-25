# Performance

What meshsearch's queries, construction and memory cost, measured against
scipy's `cKDTree` and scikit-learn's `BallTree` from Python, and what the cell
side changes. The advice drawn from these figures is in the
[guide](guide.md#choosing-the-cell-side); this page holds the measurements.

## How it was measured

| | |
|---|---|
| machine | Apple M1 Pro, 10 cores, 16 GiB, macOS 14.5 |
| compiler | Apple clang 15, CMake Release build (`-O3`) |
| Python | 3.14.7, numpy 2.5.5, scipy 1.18.1, scikit-learn 1.9.1 |

Every row is one library at one size, run in a process of its own, on one
thread, with the machine otherwise idle. A query time is the median of three
runs, in microseconds per query point. Each run draws 10,000 new query points,
in random order, and every library is given the same draws, so no run finds
the cells or tree nodes it visits still in the processor's cache from the run
before. *Per call* is one Python call per query point, so it includes the cost
of the call itself; *batched* is one call for all 10,000. *Memory* is how much the process's
resident size grew while the structure was built, which includes memory the
build used temporarily and the allocator kept. Every structure is built with
its default parameters; meshsearch with explicit limits, the unit cube.
Differences under about 10 per cent are within the noise between runs.

## The data

The *mean separation* *s* of *N* points in a volume *V* is the cube root of *V*/*N*: the
typical distance between neighbours if the points were spread evenly. Cell
sides and radii below are given in units of *s* for the whole box. The radius
query uses a radius of 2 *s*, and the k-nearest query k = 10.

- **Uniform:** points uniform in the unit cube; query points likewise. A
  ball of radius 2 *s* holds about 33 objects.
- **Clustered:** a Thomas process in the unit cube. Nine tenths of the points
  lie in clusters of about 1000, each a Gaussian of width *s* around a centre
  drawn uniformly; the rest are uniform. Query points are fresh draws from the
  same clusters, so they fall where the objects are dense: a ball of radius
  2 *s* holds about 400 objects, twelve times as many as for uniform data.

## Random order and spatial order

The tables describe two realistic ways of asking queries:

- **Fresh query points in random order**, the tables against `cKDTree` and
  `BallTree` and the periodic ones. Consecutive queries fall far apart, so
  each reads memory the one before did not.
- **Query points in spatial order**, the [large-batch](#large-batches) table.
  meshsearch answers a batch of 10⁵ or more query points in an order that
  follows the cells, which is the same as being handed them in spatial order;
  its *input order* columns are the same points in random order.

At one mean separation the spatial order makes queries 1.6 to 2.6 times
faster; so if you can, query in spatial order, or pass large batches and let
meshsearch order them.

## In short

- **Construction** is 1.7 to 4.7 times faster than either tree, at every size
  and on both data sets.
- **Batched, at a cell side of one mean separation**, meshsearch is ahead of or
  level with `cKDTree` on every query kind, uniform and clustered. On
  clustered data its radius query is 3.5 to 6.6 times faster than
  `cKDTree`'s at ten million points, at either cell side.
- **Per call**, a meshsearch query for the nearest object costs about 0.5 to
  1.1 µs more than the same query batched, and a radius query up to about
  2.4 µs more; a `cKDTree` call for the nearest object costs 8 to 10 µs.
  meshsearch is ahead per call on the nearest and the ten nearest in every
  row, and ahead of or level with it on the radius query in every row.
- **At four mean separations**, batched, the radius query is 4.0 to 6.8 times
  faster than `cKDTree`'s, but the nearest object is 1.1 to 4.8 times
  slower.
- **Periodic boxes** cost up to about 50 per cent over the same grid without
  periodicity, batched; at one mean separation meshsearch stays ahead of
  `cKDTree` with its periodic `boxsize` on every query kind.
- **Large batches**, answered in cell order, make meshsearch 1.8 to 4.3 times
  faster than `cKDTree` on every query kind at one mean separation.
- **Memory** is the price. From a million points up, meshsearch's build grows
  the process by about 1.7 times as much as `cKDTree`'s at four mean
  separations, and 3 to 5 times at one; `BallTree` uses less than either
  at ten million.

## Against cKDTree and BallTree

Every figure in these and the periodic tables comes from one run. The trees
have no cell side, so their columns are the same in the two tables of each
data set.

### Uniform, cell side 1 *s*

| points | | meshsearch | cKDTree | BallTree |
|---|---|---|---|---|
| **100,000** | construction (seconds) | 0.015 | 0.027 | 0.027 |
|  | memory (MiB) | 23 | 28 | 87 |
|  | nearest, per call | 1.02 | 8.03 | 31.98 |
|  | nearest, batched | 0.42 | 0.80 | 3.40 |
|  | k = 10, per call | 3.01 | 10.89 | 35.53 |
|  | k = 10, batched | 1.79 | 1.97 | 6.85 |
|  | radius, per call | 5.16 | 6.07 | 50.64 |
|  | radius, batched | 3.81 | 4.28 | 3.30 |
|  | objects within the radius | 31.9 | 31.9 | 31.9 |
| **1,000,000** | construction (seconds) | 0.190 | 0.367 | 0.369 |
|  | memory (MiB) | 266 | 52 | 96 |
|  | nearest, per call | 1.69 | 9.30 | 49.29 |
|  | nearest, batched | 0.95 | 1.55 | 17.52 |
|  | k = 10, per call | 5.16 | 13.02 | 60.83 |
|  | k = 10, batched | 3.35 | 3.57 | 28.90 |
|  | radius, per call | 8.96 | 9.31 | 750.29 |
|  | radius, batched | 7.18 | 7.36 | 9.08 |
|  | objects within the radius | 32.7 | 32.7 | 32.7 |
| **10,000,000** | construction (seconds) | 2.670 | 6.225 | 6.838 |
|  | memory (MiB) | 1,573 | 399 | 176 |
|  | nearest, per call | 2.28 | 9.88 | 97.06 |
|  | nearest, batched | 1.14 | 2.89 | 63.76 |
|  | k = 10, per call | 7.11 | 14.22 | 123.80 |
|  | k = 10, batched | 5.24 | 6.12 | 89.53 |
|  | radius, per call | 11.89 | 12.69 | 6,881.01 |
|  | radius, batched | 9.51 | 9.82 | 18.06 |
|  | objects within the radius | 33.2 | 33.2 | 33.2 |

### Uniform, cell side 4 *s*

| points | | meshsearch | cKDTree | BallTree |
|---|---|---|---|---|
| **100,000** | construction (seconds) | 0.012 | 0.027 | 0.027 |
|  | memory (MiB) | 8 | 28 | 87 |
|  | nearest, per call | 2.59 | 8.03 | 31.98 |
|  | nearest, batched | 1.94 | 0.80 | 3.40 |
|  | k = 10, per call | 3.82 | 10.89 | 35.53 |
|  | k = 10, batched | 2.67 | 1.97 | 6.85 |
|  | radius, per call | 2.39 | 6.07 | 50.64 |
|  | radius, batched | 1.07 | 4.28 | 3.30 |
|  | objects within the radius | 31.9 | 31.9 | 31.9 |
| **1,000,000** | construction (seconds) | 0.135 | 0.367 | 0.369 |
|  | memory (MiB) | 92 | 52 | 96 |
|  | nearest, per call | 3.22 | 9.30 | 49.29 |
|  | nearest, batched | 2.62 | 1.55 | 17.52 |
|  | k = 10, per call | 4.85 | 13.02 | 60.83 |
|  | k = 10, batched | 3.53 | 3.57 | 28.90 |
|  | radius, per call | 3.16 | 9.31 | 750.29 |
|  | radius, batched | 1.70 | 7.36 | 9.08 |
|  | objects within the radius | 32.7 | 32.7 | 32.7 |
| **10,000,000** | construction (seconds) | 1.480 | 6.225 | 6.838 |
|  | memory (MiB) | 683 | 399 | 176 |
|  | nearest, per call | 3.81 | 9.88 | 97.06 |
|  | nearest, batched | 3.21 | 2.89 | 63.76 |
|  | k = 10, per call | 6.38 | 14.22 | 123.80 |
|  | k = 10, batched | 4.97 | 6.12 | 89.53 |
|  | radius, per call | 4.08 | 12.69 | 6,881.01 |
|  | radius, batched | 2.35 | 9.82 | 18.06 |
|  | objects within the radius | 33.2 | 33.2 | 33.2 |

### Clustered, cell side 1 *s*

| points | | meshsearch | cKDTree | BallTree |
|---|---|---|---|---|
| **100,000** | construction (seconds) | 0.015 | 0.027 | 0.026 |
|  | memory (MiB) | 19 | 23 | 87 |
|  | nearest, per call | 1.36 | 7.79 | 36.83 |
|  | nearest, batched | 0.75 | 0.78 | 8.42 |
|  | k = 10, per call | 2.91 | 10.61 | 43.59 |
|  | k = 10, batched | 1.71 | 1.92 | 15.06 |
|  | radius, per call | 6.96 | 19.58 | 63.74 |
|  | radius, batched | 5.45 | 27.02 | 9.24 |
|  | objects within the radius | 385.0 | 385.0 | 385.0 |
| **1,000,000** | construction (seconds) | 0.183 | 0.359 | 0.362 |
|  | memory (MiB) | 194 | 49 | 95 |
|  | nearest, per call | 2.01 | 9.01 | 80.33 |
|  | nearest, batched | 1.25 | 1.77 | 47.00 |
|  | k = 10, per call | 4.30 | 12.08 | 101.68 |
|  | k = 10, batched | 3.02 | 3.92 | 66.23 |
|  | radius, per call | 10.80 | 29.72 | 758.65 |
|  | radius, batched | 9.03 | 36.80 | 24.57 |
|  | objects within the radius | 410.2 | 410.2 | 410.2 |
| **10,000,000** | construction (seconds) | 2.372 | 5.986 | 6.593 |
|  | memory (MiB) | 1,054 | 335 | 178 |
|  | nearest, per call | 2.78 | 10.08 | 187.03 |
|  | nearest, batched | 2.04 | 2.94 | 145.98 |
|  | k = 10, per call | 5.26 | 13.93 | 237.07 |
|  | k = 10, batched | 3.45 | 5.32 | 199.38 |
|  | radius, per call | 14.51 | 37.43 | 6,854.37 |
|  | radius, batched | 12.66 | 44.70 | 44.10 |
|  | objects within the radius | 414.6 | 414.6 | 414.6 |

### Clustered, cell side 4 *s*

| points | | meshsearch | cKDTree | BallTree |
|---|---|---|---|---|
| **100,000** | construction (seconds) | 0.012 | 0.027 | 0.026 |
|  | memory (MiB) | 5 | 23 | 87 |
|  | nearest, per call | 4.40 | 7.79 | 36.83 |
|  | nearest, batched | 3.76 | 0.78 | 8.42 |
|  | k = 10, per call | 4.38 | 10.61 | 43.59 |
|  | k = 10, batched | 3.17 | 1.92 | 15.06 |
|  | radius, per call | 5.95 | 19.58 | 63.74 |
|  | radius, batched | 4.37 | 27.02 | 9.24 |
|  | objects within the radius | 385.0 | 385.0 | 385.0 |
| **1,000,000** | construction (seconds) | 0.131 | 0.359 | 0.362 |
|  | memory (MiB) | 85 | 49 | 95 |
|  | nearest, per call | 5.11 | 9.01 | 80.33 |
|  | nearest, batched | 4.65 | 1.77 | 47.00 |
|  | k = 10, per call | 5.34 | 12.08 | 101.68 |
|  | k = 10, batched | 4.01 | 3.92 | 66.23 |
|  | radius, per call | 6.92 | 29.72 | 758.65 |
|  | radius, batched | 5.40 | 36.80 | 24.57 |
|  | objects within the radius | 410.2 | 410.2 | 410.2 |
| **10,000,000** | construction (seconds) | 1.393 | 5.986 | 6.593 |
|  | memory (MiB) | 584 | 335 | 178 |
|  | nearest, per call | 6.36 | 10.08 | 187.03 |
|  | nearest, batched | 5.72 | 2.94 | 145.98 |
|  | k = 10, per call | 6.87 | 13.93 | 237.07 |
|  | k = 10, batched | 5.99 | 5.32 | 199.38 |
|  | radius, per call | 8.18 | 37.43 | 6,854.37 |
|  | radius, batched | 6.73 | 44.70 | 44.10 |
|  | objects within the radius | 414.6 | 414.6 | 414.6 |

## Periodic boxes

All three axes periodic, against `cKDTree` with `boxsize=1`; `BallTree` has no
periodic mode. The cell side must fit a whole number of times into the
period, so at ten million points the grid used 0.004630 for a requested
0.004642 (1 *s*), and 0.018519 for 0.018566 (4 *s*). The non-periodic tables
above are the reference for the extra cost.

### Uniform

| points | | meshsearch, 1 *s* | meshsearch, 4 *s* | cKDTree, boxsize |
|---|---|---|---|---|
| **1,000,000** | construction (seconds) | 0.169 | 0.134 | 0.414 |
|  | memory (MiB) | 106 | 77 | 50 |
|  | nearest, per call | 1.97 | 3.49 | 9.77 |
|  | nearest, batched | 1.32 | 2.55 | 2.53 |
|  | k = 10, per call | 6.54 | 5.21 | 14.72 |
|  | k = 10, batched | 5.03 | 3.78 | 5.85 |
|  | radius, per call | 11.31 | 3.44 | 12.55 |
|  | radius, batched | 9.33 | 1.88 | 11.02 |
|  | objects within the radius | 33.5 | 33.5 | 33.5 |
| **10,000,000** | construction (seconds) | 1.815 | 1.406 | 6.149 |
|  | memory (MiB) | 969 | 686 | 428 |
|  | nearest, per call | 2.21 | 3.97 | 11.51 |
|  | nearest, batched | 1.54 | 3.31 | 3.50 |
|  | k = 10, per call | 7.11 | 6.34 | 17.58 |
|  | k = 10, batched | 6.16 | 4.87 | 8.12 |
|  | radius, per call | 13.35 | 3.87 | 17.88 |
|  | radius, batched | 11.85 | 2.61 | 15.93 |
|  | objects within the radius | 33.5 | 33.5 | 33.5 |

### Clustered

| points | | meshsearch, 1 *s* | meshsearch, 4 *s* | cKDTree, boxsize |
|---|---|---|---|---|
| **1,000,000** | construction (seconds) | 0.154 | 0.135 | 0.367 |
|  | memory (MiB) | 91 | 77 | 51 |
|  | nearest, per call | 2.18 | 5.53 | 10.13 |
|  | nearest, batched | 1.53 | 4.91 | 2.46 |
|  | k = 10, per call | 4.76 | 5.60 | 14.96 |
|  | k = 10, batched | 3.46 | 4.22 | 5.58 |
|  | radius, per call | 13.06 | 7.38 | 39.17 |
|  | radius, batched | 11.09 | 5.89 | 46.43 |
|  | objects within the radius | 419.6 | 419.6 | 419.6 |
| **10,000,000** | construction (seconds) | 1.878 | 1.381 | 6.117 |
|  | memory (MiB) | 864 | 552 | 363 |
|  | nearest, per call | 2.68 | 6.66 | 11.06 |
|  | nearest, batched | 2.37 | 5.91 | 3.42 |
|  | k = 10, per call | 5.44 | 7.40 | 16.29 |
|  | k = 10, batched | 4.61 | 6.05 | 7.99 |
|  | radius, per call | 17.79 | 8.77 | 46.89 |
|  | radius, batched | 16.23 | 6.80 | 58.37 |
|  | objects within the radius | 418.7 | 418.7 | 418.7 |

## Per call or batched

A separate measurement of the cost of one Python call, on a million uniform
points with a cell side of one mean separation: the median of five runs of
20,000 calls each, in µs per call.

| query | per call |
|---|---|
| nearest, three Python floats | 1.06 |
| nearest, a numpy float64 row unpacked | 1.49 |
| k = 10 | 4.38 |
| radius 2 *s* | 8.01 |
| `contains` | 0.05 |
| `is_alive` | 0.04 |

With scipy and scikit-learn also imported into the process, the figures are
the same within 10 per cent. The calls repeat a few query points, so they
measure the call itself; with fresh query points in random order a call for
the nearest object took 1.69 µs on the same grid, and 0.95 µs per query
batched (first table above).

## Large batches

A batch of 10⁵ query points or more is answered internally in an order that
follows the cells, and the results are then returned in input order. On a
uniform grid, one call of 10⁵ or 10⁶ query points, new ones in random order
for each of the three runs; the *input order* columns are the same queries
asked in batches small enough (50,000) to be answered in the order given, so
they show what random order costs. µs per query.

| grid | queries | query | meshsearch, 1 *s* | input order, 1 *s* | meshsearch, 4 *s* | input order, 4 *s* | cKDTree |
|---|---|---|---|---|---|---|---|
| 1,000,000 | 100,000 | nearest | 0.44 | 1.02 | 2.15 | 2.54 | 1.81 |
| 1,000,000 | 100,000 | k = 10 | 1.89 | 3.95 | 2.92 | 3.60 | 4.03 |
| 1,000,000 | 100,000 | radius | 3.88 | 7.50 | 1.21 | 1.60 | 7.60 |
| 1,000,000 | 1,000,000 | nearest | 0.53 | 0.98 | 2.25 | 2.52 | 1.79 |
| 1,000,000 | 1,000,000 | k = 10 | 1.89 | 3.95 | 3.00 | 3.60 | 3.95 |
| 1,000,000 | 1,000,000 | radius | 3.77 | 7.38 | 1.34 | 1.69 | 7.49 |
| 10,000,000 | 100,000 | nearest | 0.89 | 1.42 | 2.45 | 3.48 | 2.46 |
| 10,000,000 | 100,000 | k = 10 | 2.93 | 5.68 | 3.32 | 5.27 | 5.26 |
| 10,000,000 | 100,000 | radius | 5.07 | 10.46 | 1.53 | 2.31 | 9.83 |
| 10,000,000 | 1,000,000 | nearest | 0.58 | 1.40 | 2.43 | 3.19 | 2.51 |
| 10,000,000 | 1,000,000 | k = 10 | 2.08 | 5.48 | 3.22 | 4.80 | 5.28 |
| 10,000,000 | 1,000,000 | radius | 4.24 | 10.57 | 1.40 | 2.33 | 9.90 |

The cell order is worth 1.6 to 2.6 times at one mean separation and 1.1 to 1.7
at four, about the same at 10⁵ as at 10⁶ queries. On a large batch at one mean
separation meshsearch is 1.8 to 4.3 times faster than `cKDTree` on every query
kind, and 5.6 to 7.1 times on the radius query at four.

## Cell side

Batched query times at four cell sides, with how the objects fill the cells:
the share of cells left empty, the mean count in the cells that are not, and
the 99th percentile of the count over all cells. The radius stays 2 *s*, so
every row answers the same question. The queries are drawn as for the other
tables, 10,000 fresh points per run; the whole sweep takes under a minute, so
it was run five times and each figure is the median of the five.

### Uniform, a million points

| cell side | cells | objects per cell | empty cells | per non-empty cell | 99th percentile | construction (seconds) | memory (MiB) | nearest | k = 10 | radius |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 *s* | 1,000,000 | 1.0 | 36.8 % | 1.6 | 4 | 0.192 | 216 | 0.98 | 3.89 | 7.03 |
| 2 *s* | 125,000 | 8.0 | 0.0 % | 8.0 | 15 | 0.144 | 101 | 0.90 | 2.23 | 2.38 |
| 3 *s* | 39,304 | 25.4 | 0.0 % | 25.4 | 39 | 0.137 | 88 | 1.45 | 2.78 | 1.57 |
| 4 *s* | 15,625 | 64.0 | 0.0 % | 64.0 | 83 | 0.136 | 92 | 2.55 | 3.52 | 1.64 |

### Uniform, ten million points

| cell side | cells | objects per cell | empty cells | per non-empty cell | 99th percentile | construction (seconds) | memory (MiB) | nearest | k = 10 | radius |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 *s* | 10,077,696 | 1.0 | 37.1 % | 1.6 | 4 | 2.300 | 2,165 | 1.30 | 5.42 | 10.27 |
| 2 *s* | 1,259,712 | 7.9 | 0.0 % | 7.9 | 15 | 1.704 | 850 | 1.70 | 3.69 | 4.42 |
| 3 *s* | 373,248 | 26.8 | 0.0 % | 26.8 | 40 | 1.530 | 719 | 2.15 | 4.25 | 2.56 |
| 4 *s* | 157,464 | 63.5 | 0.0 % | 63.5 | 83 | 1.451 | 697 | 3.34 | 5.02 | 2.51 |

### Clustered, a million points

| cell side | cells | objects per cell | empty cells | per non-empty cell | 99th percentile | construction (seconds) | memory (MiB) | nearest | k = 10 | radius |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 *s* | 1,000,000 | 1.0 | 80.1 % | 5.0 | 24 | 0.186 | 221 | 1.20 | 2.79 | 8.62 |
| 2 *s* | 125,000 | 8.0 | 34.7 % | 12.3 | 167 | 0.140 | 98 | 2.18 | 3.00 | 5.71 |
| 3 *s* | 39,304 | 25.4 | 5.7 % | 27.0 | 414 | 0.136 | 89 | 3.17 | 3.38 | 5.00 |
| 4 *s* | 15,625 | 64.0 | 0.1 % | 64.0 | 749 | 0.136 | 77 | 4.70 | 4.03 | 5.41 |

### Clustered, ten million points

| cell side | cells | objects per cell | empty cells | per non-empty cell | 99th percentile | construction (seconds) | memory (MiB) | nearest | k = 10 | radius |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 *s* | 10,077,696 | 1.0 | 80.2 % | 5.0 | 24 | 2.198 | 2,181 | 1.85 | 3.83 | 11.90 |
| 2 *s* | 1,259,712 | 7.9 | 34.9 % | 12.2 | 165 | 1.597 | 737 | 3.06 | 4.40 | 7.65 |
| 3 *s* | 373,248 | 26.8 | 4.4 % | 28.0 | 434 | 1.447 | 610 | 4.02 | 4.69 | 5.95 |
| 4 *s* | 157,464 | 63.5 | 0.1 % | 63.6 | 730 | 1.386 | 578 | 5.49 | 5.47 | 6.20 |

On uniform data the nearest object is fastest at one to two mean separations
(level at a million points, one ahead by about 30 per cent at ten million),
and 2.6 times slower at four than at one. The ten nearest are fastest at two.
The radius query is fastest at three, level with four at ten million points,
and four to four and a half times slower at one. On the clustered set the
nearest and the ten nearest are fastest at one, where four fifths of the cells
are empty, and slow down as the cells grow: at four, one cell in a hundred
holds over ten times the average, and the nearest object is three to four
times slower than at one. The radius query is fastest at three.

At two mean separations the cell side equals the radius, and that is a step:
a ball whose radius is at least the cell side costs more than one just under
it. On a million uniform points, in a separate measurement (median of five
runs of 10,000 fresh queries), the radius query took 2.27 µs at a cell side
of exactly 2 *s* and 1.55 µs at 2.02 *s*; at 3 *s* it took 1.50.

## Memory

A grid keeps 37 bytes per object (the coordinates, 24, and the bookkeeping
that maps indices to storage, 13) and at least 112 bytes per cell (16 for the
cell, and 96 for its eight entries in a table of cell offsets; about a quarter
of that on a grid periodic along all three axes). At ten million uniform points
and four mean separations that is 353 MiB for the objects and about 17 MiB for
the 157,464 cells. The build needs, besides, a temporary cell index of 8 bytes
per object, and from Python a copy of the three coordinate arrays, 24 bytes per
object; both are freed when the grid is built, but the allocator kept them in
the process, which is why the table above reports about 700 MiB for that grid.
A C++ program building the same grid grew by 455 MiB. The grid is built
compact, with no room reserved for later additions.

At one mean separation there are as many cells as objects, so the cells cost
more than the objects; at ten million points the measured growth varied
between identical runs from 1.0 to 2.1 GiB, on either data set. The
four-mean-separation figures repeated within 2 per cent.

## Not measured

Threads; additions and removals; boxes much longer along one axis than the
others; clustering of other strengths or shapes; radii other than two mean
separations; the batched `count_objects` and `contains` against the trees; and machines other
than the one above.

## Reproducing

The scripts are in `benchmarks/`; see `benchmarks/README.md` for what each
measures and its options. From a source checkout:

```
pip install -C cmake.build-type=Release .
pip install scipy scikit-learn psutil
python benchmarks/benchmark.py
python benchmarks/benchmark.py --periodic --sizes 1000000 10000000
python benchmarks/benchmark.py --large-batch --sizes 1000000 10000000
python benchmarks/benchmark_cellsize.py --sweep
```

Each prints its rows as JSON, from which the tables above were made.
The first three took 17 minutes together on the machine above, most of it in
`BallTree`'s per-call radius queries at ten million points.
