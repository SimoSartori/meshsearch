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
runs of the same 1000 query points, in microseconds per query point. *Per
call* is one Python call per query point, so it includes the cost of the call
itself; *batched* is one call for all 1000. *Memory* is how much the process's
resident size grew while the structure was built, which includes memory the
build used temporarily and the allocator kept. Every structure is built with
its default parameters; meshsearch with explicit limits, the unit cube.
Differences under about 10 per cent are within the noise between runs.

Because the same 1000 query points are run three times, the cells or tree
nodes they visit can still be in the processor's cache from the first run.
The figures therefore describe queries over a working set that fits in cache,
for every library alike. Larger batches of distinct query points cost more per
query, most of all at fine cell sides.

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

## In short

- **Construction** is 1.6 to 4.7 times faster than either tree, at every size
  and on both data sets.
- **Batched, at a cell side of one mean separation**, meshsearch is ahead of or
  level with `cKDTree` on every query kind, uniform and clustered. On
  clustered data its radius query is 3.6 to 6.3 times faster than
  `cKDTree`'s at ten million points.
- **Per call**, a meshsearch query for the nearest object costs about 0.6 µs
  more than the same query batched, and a radius query up to about 3 µs more;
  a `cKDTree` call for the nearest object costs 8 to 9 µs. meshsearch is ahead
  per call on the nearest and the ten nearest in every row, and on the radius
  query in every row except uniform data at one mean separation, where it is
  level.
- **At four mean separations**, batched, the radius query is 3.7 to 6.3 times
  faster than `cKDTree`'s, but the nearest object is 1.8 to 4.7 times
  slower.
- **Periodic boxes** cost up to about 45 per cent over the same grid without
  periodicity, batched; at one mean separation meshsearch stays ahead of
  `cKDTree` with its periodic `boxsize` on every query kind.
- **Memory** is the price. From a million points up, meshsearch's build grows
  the process by 1.6 to 2.1 times as much as `cKDTree`'s at four mean
  separations, and 2.8 to 6.2 times at one; `BallTree` uses less than either
  at ten million.

## Against cKDTree and BallTree

meshsearch's per-call times, construction and memory come from a second run
with the same scripts and machine; the rest from the first. The trees have no
cell side, so their columns are the same in the two tables of each data set.

### Uniform, cell side 1 *s*

| points | | meshsearch | cKDTree | BallTree |
|---|---|---|---|---|
| **100,000** | construction (seconds) | 0.017 | 0.029 | 0.028 |
|  | memory (MiB) | 24 | 26 | 86 |
|  | nearest, per call | 1.32 | 8.53 | 34.22 |
|  | nearest, batched | 0.55 | 0.82 | 3.87 |
|  | k = 10, per call | 3.02 | 10.31 | 37.46 |
|  | k = 10, batched | 2.22 | 2.01 | 7.79 |
|  | radius, per call | 5.48 | 6.23 | 51.11 |
|  | radius, batched | 3.98 | 4.20 | 3.61 |
|  | objects within the radius | 32.1 | 32.1 | 32.1 |
| **1,000,000** | construction (seconds) | 0.191 | 0.378 | 0.378 |
|  | memory (MiB) | 225 | 53 | 96 |
|  | nearest, per call | 1.13 | 8.73 | 46.83 |
|  | nearest, batched | 0.47 | 1.21 | 17.23 |
|  | k = 10, per call | 3.95 | 11.17 | 59.33 |
|  | k = 10, batched | 2.74 | 2.99 | 28.45 |
|  | radius, per call | 8.81 | 8.27 | 882.57 |
|  | radius, batched | 6.49 | 6.42 | 10.05 |
|  | objects within the radius | 32.9 | 32.9 | 32.9 |
| **10,000,000** | construction (seconds) | 2.639 | 5.982 | 6.691 |
|  | memory (MiB) | 1,103 | 398 | 177 |
|  | nearest, per call | 1.14 | 9.28 | 84.95 |
|  | nearest, batched | 0.49 | 1.53 | 52.77 |
|  | k = 10, per call | 4.97 | 12.70 | 118.89 |
|  | k = 10, batched | 3.18 | 3.71 | 75.37 |
|  | radius, per call | 11.13 | 11.35 | 8,219.10 |
|  | radius, batched | 7.80 | 9.42 | 25.55 |
|  | objects within the radius | 33.2 | 33.2 | 33.2 |

### Uniform, cell side 4 *s*

| points | | meshsearch | cKDTree | BallTree |
|---|---|---|---|---|
| **100,000** | construction (seconds) | 0.012 | 0.029 | 0.028 |
|  | memory (MiB) | 9 | 26 | 86 |
|  | nearest, per call | 2.75 | 8.53 | 34.22 |
|  | nearest, batched | 2.03 | 0.82 | 3.87 |
|  | k = 10, per call | 3.98 | 10.31 | 37.46 |
|  | k = 10, batched | 2.77 | 2.01 | 7.79 |
|  | radius, per call | 2.35 | 6.23 | 51.11 |
|  | radius, batched | 1.13 | 4.20 | 3.61 |
|  | objects within the radius | 32.1 | 32.1 | 32.1 |
| **1,000,000** | construction (seconds) | 0.137 | 0.378 | 0.378 |
|  | memory (MiB) | 85 | 53 | 96 |
|  | nearest, per call | 3.19 | 8.73 | 46.83 |
|  | nearest, batched | 2.51 | 1.21 | 17.23 |
|  | k = 10, per call | 4.85 | 11.17 | 59.33 |
|  | k = 10, batched | 3.36 | 2.99 | 28.45 |
|  | radius, per call | 2.73 | 8.27 | 882.57 |
|  | radius, batched | 1.56 | 6.42 | 10.05 |
|  | objects within the radius | 32.9 | 32.9 | 32.9 |
| **10,000,000** | construction (seconds) | 1.430 | 5.982 | 6.691 |
|  | memory (MiB) | 704 | 398 | 177 |
|  | nearest, per call | 3.77 | 9.28 | 84.95 |
|  | nearest, batched | 2.78 | 1.53 | 52.77 |
|  | k = 10, per call | 6.28 | 12.70 | 118.89 |
|  | k = 10, batched | 4.34 | 3.71 | 75.37 |
|  | radius, per call | 3.43 | 11.35 | 8,219.10 |
|  | radius, batched | 1.73 | 9.42 | 25.55 |
|  | objects within the radius | 33.2 | 33.2 | 33.2 |

### Clustered, cell side 1 *s*

| points | | meshsearch | cKDTree | BallTree |
|---|---|---|---|---|
| **100,000** | construction (seconds) | 0.015 | 0.028 | 0.027 |
|  | memory (MiB) | 22 | 26 | 85 |
|  | nearest, per call | 1.63 | 8.22 | 39.96 |
|  | nearest, batched | 0.78 | 0.83 | 8.84 |
|  | k = 10, per call | 2.80 | 9.94 | 43.98 |
|  | k = 10, batched | 1.70 | 2.03 | 15.79 |
|  | radius, per call | 6.59 | 17.37 | 58.94 |
|  | radius, batched | 5.11 | 24.75 | 9.16 |
|  | objects within the radius | 389.0 | 389.0 | 389.0 |
| **1,000,000** | construction (seconds) | 0.170 | 0.381 | 0.361 |
|  | memory (MiB) | 193 | 47 | 95 |
|  | nearest, per call | 1.60 | 8.37 | 74.42 |
|  | nearest, batched | 1.07 | 1.36 | 41.22 |
|  | k = 10, per call | 3.37 | 11.12 | 95.05 |
|  | k = 10, batched | 2.30 | 3.91 | 61.23 |
|  | radius, per call | 10.08 | 25.98 | 913.13 |
|  | radius, batched | 8.20 | 33.72 | 23.82 |
|  | objects within the radius | 407.7 | 407.7 | 407.7 |
| **10,000,000** | construction (seconds) | 2.209 | 5.971 | 6.356 |
|  | memory (MiB) | 2,075 | 334 | 176 |
|  | nearest, per call | 1.88 | 9.12 | 177.16 |
|  | nearest, batched | 1.12 | 1.59 | 128.26 |
|  | k = 10, per call | 3.81 | 13.64 | 205.55 |
|  | k = 10, batched | 2.57 | 4.07 | 172.58 |
|  | radius, per call | 12.19 | 30.84 | 7,836.61 |
|  | radius, batched | 10.46 | 37.90 | 50.27 |
|  | objects within the radius | 416.0 | 416.0 | 416.0 |

### Clustered, cell side 4 *s*

| points | | meshsearch | cKDTree | BallTree |
|---|---|---|---|---|
| **100,000** | construction (seconds) | 0.012 | 0.028 | 0.027 |
|  | memory (MiB) | 4 | 26 | 85 |
|  | nearest, per call | 4.65 | 8.22 | 39.96 |
|  | nearest, batched | 3.90 | 0.83 | 8.84 |
|  | k = 10, per call | 4.73 | 9.94 | 43.98 |
|  | k = 10, batched | 3.17 | 2.03 | 15.79 |
|  | radius, per call | 6.61 | 17.37 | 58.94 |
|  | radius, batched | 4.54 | 24.75 | 9.16 |
|  | objects within the radius | 389.0 | 389.0 | 389.0 |
| **1,000,000** | construction (seconds) | 0.130 | 0.381 | 0.361 |
|  | memory (MiB) | 97 | 47 | 95 |
|  | nearest, per call | 5.31 | 8.37 | 74.42 |
|  | nearest, batched | 4.67 | 1.36 | 41.22 |
|  | k = 10, per call | 5.19 | 11.12 | 95.05 |
|  | k = 10, batched | 4.17 | 3.91 | 61.23 |
|  | radius, per call | 6.52 | 25.98 | 913.13 |
|  | radius, batched | 5.45 | 33.72 | 23.82 |
|  | objects within the radius | 407.7 | 407.7 | 407.7 |
| **10,000,000** | construction (seconds) | 1.394 | 5.971 | 6.356 |
|  | memory (MiB) | 583 | 334 | 176 |
|  | nearest, per call | 5.91 | 9.12 | 177.16 |
|  | nearest, batched | 5.02 | 1.59 | 128.26 |
|  | k = 10, per call | 6.16 | 13.64 | 205.55 |
|  | k = 10, batched | 4.87 | 4.07 | 172.58 |
|  | radius, per call | 7.39 | 30.84 | 7,836.61 |
|  | radius, batched | 5.99 | 37.90 | 50.27 |
|  | objects within the radius | 416.0 | 416.0 | 416.0 |

## Periodic boxes

All three axes periodic, against `cKDTree` with `boxsize=1`; `BallTree` has no
periodic mode. The cell side must fit a whole number of times into the
period, so at ten million points the grid used 0.004630 for a requested
0.004642 (1 *s*), and 0.018519 for 0.018566 (4 *s*). The non-periodic tables
above are the reference for the extra cost.

### Uniform

| points | | meshsearch, 1 *s* | meshsearch, 4 *s* | cKDTree, boxsize |
|---|---|---|---|---|
| **1,000,000** | construction (seconds) | 0.160 | 0.130 | 0.391 |
|  | memory (MiB) | 128 | 90 | 51 |
|  | nearest, per call | 1.27 | 3.36 | 9.80 |
|  | nearest, batched | 0.56 | 2.79 | 1.75 |
|  | k = 10, per call | 4.72 | 4.84 | 13.18 |
|  | k = 10, batched | 3.36 | 3.61 | 4.46 |
|  | radius, per call | 11.36 | 2.98 | 11.92 |
|  | radius, batched | 7.30 | 1.89 | 11.15 |
|  | objects within the radius | 33.6 | 33.6 | 33.6 |
| **10,000,000** | construction (seconds) | 1.827 | 1.472 | 6.083 |
|  | memory (MiB) | 985 | 672 | 427 |
|  | nearest, per call | 1.23 | 3.70 | 9.84 |
|  | nearest, batched | 0.54 | 3.07 | 2.01 |
|  | k = 10, per call | 4.87 | 6.61 | 15.01 |
|  | k = 10, batched | 4.51 | 4.41 | 5.34 |
|  | radius, per call | 11.26 | 4.15 | 15.76 |
|  | radius, batched | 10.21 | 1.96 | 13.76 |
|  | objects within the radius | 33.5 | 33.5 | 33.5 |

### Clustered

| points | | meshsearch, 1 *s* | meshsearch, 4 *s* | cKDTree, boxsize |
|---|---|---|---|---|
| **1,000,000** | construction (seconds) | 0.155 | 0.137 | 0.387 |
|  | memory (MiB) | 119 | 85 | 45 |
|  | nearest, per call | 1.73 | 5.37 | 9.50 |
|  | nearest, batched | 1.08 | 4.91 | 1.69 |
|  | k = 10, per call | 3.77 | 5.72 | 13.18 |
|  | k = 10, batched | 2.55 | 4.27 | 4.29 |
|  | radius, per call | 12.87 | 7.34 | 33.95 |
|  | radius, batched | 10.44 | 5.72 | 42.77 |
|  | objects within the radius | 417.3 | 417.3 | 417.3 |
| **10,000,000** | construction (seconds) | 1.790 | 1.407 | 5.875 |
|  | memory (MiB) | 919 | 564 | 363 |
|  | nearest, per call | 1.93 | 6.58 | 9.94 |
|  | nearest, batched | 1.39 | 5.75 | 2.02 |
|  | k = 10, per call | 3.65 | 7.00 | 15.35 |
|  | k = 10, batched | 3.12 | 5.31 | 5.90 |
|  | radius, per call | 13.98 | 8.16 | 43.61 |
|  | radius, batched | 14.24 | 6.53 | 51.50 |
|  | objects within the radius | 418.9 | 418.9 | 418.9 |

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
the same within 10 per cent. Batched on the same grid, the nearest object takes
0.47 µs per query (first table above).

## Large batches

A batch of many query points is answered internally in an order that follows
the cells, and the results are then returned in input order. On a uniform grid,
one call of 10⁵ or 10⁶ query points; the *input order* columns are the same
queries asked in batches small enough to be answered in the order given. µs
per query.

| grid | queries | query | meshsearch, 1 *s* | input order, 1 *s* | meshsearch, 4 *s* | input order, 4 *s* | cKDTree |
|---|---|---|---|---|---|---|---|
| 1,000,000 | 100,000 | nearest | 0.46 | 0.99 | 2.30 | 2.57 | 1.68 |
| 1,000,000 | 100,000 | k = 10 | 1.84 | 3.59 | 2.99 | 3.48 | 3.75 |
| 1,000,000 | 100,000 | radius | 4.09 | 6.88 | 1.28 | 1.62 | 6.95 |
| 1,000,000 | 1,000,000 | nearest | 0.51 | 0.91 | 2.34 | 2.56 | 1.65 |
| 1,000,000 | 1,000,000 | k = 10 | 1.93 | 3.65 | 2.95 | 3.50 | 3.61 |
| 1,000,000 | 1,000,000 | radius | 3.82 | 6.87 | 1.34 | 1.56 | 6.82 |
| 10,000,000 | 100,000 | nearest | 0.79 | 1.17 | 2.49 | 3.23 | 2.48 |
| 10,000,000 | 100,000 | k = 10 | 2.67 | 4.95 | 3.21 | 4.70 | 5.22 |
| 10,000,000 | 100,000 | radius | 4.95 | 9.38 | 1.48 | 2.53 | 9.83 |
| 10,000,000 | 1,000,000 | nearest | 0.57 | 1.23 | 2.38 | 3.11 | 2.52 |
| 10,000,000 | 1,000,000 | k = 10 | 2.07 | 5.04 | 3.02 | 4.58 | 5.21 |
| 10,000,000 | 1,000,000 | radius | 4.34 | 9.34 | 1.36 | 2.32 | 9.75 |

The cell order is worth 1.5 to 2.4 times at one mean separation and 1.1 to 1.7
at four, about the same at 10⁵ as at 10⁶ queries. On a large batch at one mean
separation meshsearch is 1.7 to 4.4 times faster than `cKDTree` on every query
kind, and 5 to 7 times on the radius query at four.

## Cell side

Batched query times at four cell sides, with how the objects fill the cells:
the share of cells left empty, the mean count in the cells that are not, and
the 99th percentile of the count over all cells. The radius stays 2 *s*, so
every row answers the same question.

### Uniform, a million points

| cell side | cells | objects per cell | empty cells | per non-empty cell | 99th percentile | construction (seconds) | memory (MiB) | nearest | k = 10 | radius |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 *s* | 1,000,000 | 1.0 | 36.8 % | 1.6 | 4 | 0.193 | 233 | 0.54 | 2.89 | 6.45 |
| 2 *s* | 125,000 | 8.0 | 0.0 % | 8.0 | 15 | 0.140 | 93 | 0.75 | 1.88 | 2.11 |
| 3 *s* | 39,304 | 25.4 | 0.0 % | 25.4 | 39 | 0.146 | 81 | 1.46 | 2.88 | 1.32 |
| 4 *s* | 15,625 | 64.0 | 0.0 % | 64.0 | 83 | 0.137 | 92 | 2.42 | 3.37 | 1.43 |

### Uniform, ten million points

| cell side | cells | objects per cell | empty cells | per non-empty cell | 99th percentile | construction (seconds) | memory (MiB) | nearest | k = 10 | radius |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 *s* | 10,077,696 | 1.0 | 37.1 % | 1.6 | 4 | 2.196 | 2,154 | 0.50 | 3.02 | 7.42 |
| 2 *s* | 1,259,712 | 7.9 | 0.0 % | 7.9 | 15 | 1.659 | 853 | 0.84 | 2.27 | 2.59 |
| 3 *s* | 373,248 | 26.8 | 0.0 % | 26.8 | 40 | 1.560 | 719 | 1.55 | 3.05 | 1.62 |
| 4 *s* | 157,464 | 63.5 | 0.0 % | 63.5 | 83 | 1.500 | 689 | 3.92 | 4.88 | 1.99 |

### Clustered, a million points

| cell side | cells | objects per cell | empty cells | per non-empty cell | 99th percentile | construction (seconds) | memory (MiB) | nearest | k = 10 | radius |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 *s* | 1,000,000 | 1.0 | 80.1 % | 5.0 | 24 | 0.195 | 216 | 1.10 | 2.46 | 9.20 |
| 2 *s* | 125,000 | 8.0 | 34.7 % | 12.3 | 167 | 0.149 | 104 | 2.22 | 2.95 | 5.31 |
| 3 *s* | 39,304 | 25.4 | 5.7 % | 27.0 | 414 | 0.139 | 92 | 3.19 | 3.34 | 4.89 |
| 4 *s* | 15,625 | 64.0 | 0.1 % | 64.0 | 749 | 0.135 | 84 | 4.67 | 3.99 | 5.35 |

### Clustered, ten million points

| cell side | cells | objects per cell | empty cells | per non-empty cell | 99th percentile | construction (seconds) | memory (MiB) | nearest | k = 10 | radius |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 *s* | 10,077,696 | 1.0 | 80.2 % | 5.0 | 24 | 2.219 | 2,147 | 1.11 | 2.57 | 10.52 |
| 2 *s* | 1,259,712 | 7.9 | 34.9 % | 12.2 | 165 | 1.570 | 731 | 2.66 | 3.52 | 7.28 |
| 3 *s* | 373,248 | 26.8 | 4.4 % | 28.0 | 434 | 1.468 | 613 | 4.08 | 4.51 | 5.92 |
| 4 *s* | 157,464 | 63.5 | 0.1 % | 63.6 | 730 | 1.386 | 576 | 5.21 | 5.11 | 6.10 |

On uniform data the nearest object is fastest at one mean separation, the ten
nearest at two and the radius query at three to four. On the clustered set the
nearest and the ten nearest are fastest at one, where four fifths of the cells
are empty, and slow down quickly as the cells grow: at four, one cell in a
hundred holds over ten times the average. The radius query is fastest at three.

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
between identical runs from 1.1 to 2.1 GiB, on either data set. The
four-mean-separation figures repeated within 2 per cent.

## Not measured

Threads; additions and removals; boxes much longer along one axis than the
others; clustering of other strengths or shapes; radii other than two mean
separations; batches of distinct query points too large to stay in cache; the
batched `count_objects` and `contains` against the trees; and machines other
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
Together they took about a quarter of an hour on the machine above, most of it
building the ten-million-point structures.
