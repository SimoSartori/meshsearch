# Performance

Three questions: how meshsearch compares to the alternatives, how to choose the
cell side, and what to expect on data unlike the data measured here.

The numbers are from an Apple M1 Pro (10 cores, 16 GiB, macOS 14.5) with
Python 3.14.7, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, and meshsearch
built with AppleClang at `-O3`. Absolute times are machine-specific; the ratios
between libraries travel better than the microseconds do. Treat a difference
under about 10 per cent as a tie.

## Against cKDTree and BallTree

Three queries are measured, named here as the Python bindings spell them and
identical to their C++ counterparts:

| in the tables | the call | what it asks |
|---|---|---|
| **nearest** | `grid.nearest_object(x, y, z)` | the one nearest object |
| **k = 10** | `grid.nearest_objects(10, x, y, z)` | the ten nearest, nearest first |
| **radius** | `grid.close_objects(x, y, z, rmax)` | every object within `rmax` |

`rmax` is twice the mean separation of the data, which holds the answer at
about 33 neighbours at every size, so one query does the same amount of work as
the data grows. cKDTree and BallTree answer the same three questions, through
`query`, `query` with `k=10`, and `query_ball_point` / `query_radius`, and
return the same neighbours.

The points are uniform in the unit cube, and the queries are 1000 uniform query
points. Query times are **microseconds for one query point**. The per-call rows
call the library once per point; the batched rows hand all 1000 points over in
a single call. Memory is the resident set the structure occupies.

meshsearch is tabulated at two cell sides, because it has one and the trees do
not. One mean separation gives about one object per cell; four gives about 64.

**Cell side of one mean separation**

| points | | meshsearch | cKDTree | BallTree |
|---|---|---|---|---|
| **1e5** | construction (s) | 0.016 | 0.029 | 0.030 |
| | memory (MiB) | 29 | 26 | 88 |
| | nearest, per call | 1.09 | 8.48 | 35.67 |
| | nearest, batched | 0.33 | 0.83 | 3.75 |
| | k = 10, per call | 5.35 | 11.40 | 39.69 |
| | k = 10, batched | 4.13 | 2.03 | 7.77 |
| | radius, per call | 5.36 | 5.97 | 49.11 |
| | radius, batched | 4.16 | 4.00 | 3.83 |
| **1e6** | construction (s) | 0.199 | 0.387 | 0.394 |
| | memory (MiB) | 299 | 53 | 96 |
| | nearest, per call | 1.33 | 9.45 | 53.37 |
| | nearest, batched | 0.42 | 1.31 | 21.67 |
| | k = 10, per call | 13.33 | 12.74 | 63.88 |
| | k = 10, batched | 11.64 | 3.58 | 34.22 |
| | radius, per call | 12.32 | 8.92 | 850.24 |
| | radius, batched | 10.86 | 7.06 | 10.91 |
| **1e7** | construction (s) | 2.808 | 5.822 | 6.326 |
| | memory (MiB) | 1,495 | 400 | 177 |
| | nearest, per call | 1.79 | 8.66 | 81.26 |
| | nearest, batched | 0.58 | 1.42 | 51.85 |
| | k = 10, per call | 21.86 | 12.56 | 103.54 |
| | k = 10, batched | 16.03 | 3.86 | 73.02 |
| | radius, per call | 17.17 | 10.78 | 9,005.79 |
| | radius, batched | 14.40 | 8.79 | 23.17 |

**Cell side of four mean separations**

| points | | meshsearch | cKDTree | BallTree |
|---|---|---|---|---|
| **1e5** | construction (s) | 0.011 | 0.029 | 0.027 |
| | memory (MiB) | 10 | 28 | 87 |
| | nearest, per call | 2.61 | 8.79 | 34.97 |
| | nearest, batched | 1.91 | 0.91 | 4.04 |
| | k = 10, per call | 12.39 | 10.58 | 39.49 |
| | k = 10, batched | 11.21 | 2.27 | 8.37 |
| | radius, per call | 3.85 | 6.09 | 48.27 |
| | radius, batched | 3.09 | 4.56 | 3.94 |
| **1e6** | construction (s) | 0.134 | 0.365 | 0.366 |
| | memory (MiB) | 123 | 54 | 97 |
| | nearest, per call | 4.63 | 8.69 | 46.85 |
| | nearest, batched | 3.39 | 1.17 | 17.75 |
| | k = 10, per call | 18.60 | 11.61 | 57.03 |
| | k = 10, batched | 17.12 | 3.28 | 26.91 |
| | radius, per call | 6.13 | 8.15 | 811.82 |
| | radius, batched | 4.48 | 6.24 | 9.55 |
| **1e7** | construction (s) | 1.644 | 6.107 | 6.610 |
| | memory (MiB) | 672 | 400 | 177 |
| | nearest, per call | 8.42 | 9.60 | 85.37 |
| | nearest, batched | 7.78 | 1.43 | 53.83 |
| | k = 10, per call | 34.34 | 13.73 | 104.41 |
| | k = 10, batched | 32.37 | 4.54 | 76.44 |
| | radius, per call | 9.71 | 11.08 | 10,016.19 |
| | radius, batched | 8.76 | 9.22 | 26.97 |

### What the numbers say

**Construction is faster everywhere** — about 2x against cKDTree at one mean
separation, 2.6x to 3.7x at four.

**Memory is the weakest column at a fine cell side and the strongest at a
coarse one.** At one mean separation meshsearch holds 1.5 GiB at 1e7 against
cKDTree's 400 MiB; at four mean separations, 672 MiB, and at 1e5 it is a third
of cKDTree's.

**Nearest neighbour, one mean separation: meshsearch wins in both calling
styles.** Per call it is 5x to 8x faster than cKDTree; batched, 0.33, 0.42 and
0.58 microseconds against 0.83, 1.31 and 1.42, so 2.4x to 3.1x faster. At four
mean separations this reverses, 2.1x to 5.4x slower batched, because each cell
holds 64 candidates to test.

**Radius, four mean separations: batched meshsearch is the fastest thing
measured.** 3.09, 4.48 and 8.76 microseconds against batched cKDTree's 4.56,
6.24 and 9.22, so 1.05x to 1.48x faster at all three sizes. At one mean
separation it is 1.0x to 1.6x slower than batched cKDTree.

**k = 10 is the query to take elsewhere.** Against batched cKDTree, meshsearch
is 2.0x to 4.2x slower at one mean separation and 4.9x to 7.1x at four. The gap
is algorithmic, so no cell side and no calling style closes it: if this query
dominates, use a tree.

**BallTree loses every row.** Never call it once per point: its per-call radius
query is 9 milliseconds at 1e7 against 23 microseconds batched.

### Per call or batched

A call from Python costs about 1 microsecond before any work happens, against
7 to 8 for scipy. That is what decides how much batching can win: a query
taking 1 microsecond gets most of its time back, one taking 15 gets under 10
per cent. In these tables batching gains 3.1x to 3.3x on nearest neighbour at
one mean separation and 1.1x to 1.4x on everything else. If a query is already
slow, batching is not what will fix it.

The batched forms exist in the Python bindings only. A C++ caller has no
boundary to cross and writes a loop.

## Choosing the cell side

The cell side is the one tuning parameter, and it is not a single trade-off:
the best choice for one query kind is not the best for another. Query times in
microseconds, with `rmax` again at twice the mean separation.

**1e6 points**

| cell side | cells | objects per cell | build (s) | memory (MiB) | nearest | k = 10 | radius |
|---|---|---|---|---|---|---|---|
| 1 mps | 1,000,000 | 1 | 0.189 | 302 | 1.21 | 11.70 | 10.88 |
| 2 mps | 125,000 | 8 | 0.135 | 103 | 1.92 | 9.01 | 8.04 |
| 3 mps | 39,304 | 25 | 0.129 | 82 | 3.05 | 14.39 | 4.41 |
| 4 mps | 15,625 | 64 | 0.130 | 92 | 4.38 | 19.67 | 6.43 |

**1e7 points**

| cell side | cells | objects per cell | build (s) | memory (MiB) | nearest | k = 10 | radius |
|---|---|---|---|---|---|---|---|
| 1 mps | 10,077,696 | 1 | 2.931 | 1,622 | 1.80 | 16.71 | 14.63 |
| 2 mps | 1,259,712 | 8 | 1.487 | 876 | 3.24 | 15.75 | 14.21 |
| 3 mps | 373,248 | 27 | 1.422 | 741 | 6.72 | 26.94 | 8.29 |
| 4 mps | 157,464 | 64 | 1.459 | 701 | 9.88 | 33.20 | 9.74 |

**Nearest neighbour is the one query that only gets worse as the cell grows,**
by a factor of 3.6 at 1e6 and 5.5 at 1e7 from one mean separation to four. Its
cost follows the objects per cell.

**k-nearest is flat in the middle and worse at the ends,** best at two mean
separations at 1e6 and near-flat from one to two at 1e7, then following the
objects per cell like nearest neighbour.

**A radius query is fastest on a coarse grid, with its best value in the
middle:** 4.41 microseconds at three mean separations against 10.88 at one, and
8.29 against 14.63 at 1e7. What decides its cost is `rmax / cellsize` rather
than either alone — **keep that ratio at or below 1** and the scan stays within
the nearest cells; above it the scan widens sharply.

**Memory falls by about half, not by the drop in cell count.** From one mean
separation to four at 1e7 it goes 1,622 to 701 MiB while the cells fall 64-fold,
because what remains is a floor that does not depend on the cell side: the
grid's copy of the coordinates and its index tables. Past about three mean
separations there is no further saving to have.

### Starting point

- **Mixed use:** two to three mean separations. Nothing is at its worst there —
  at 1e7 it costs 3.24 to 6.72 microseconds on nearest neighbour, 15.75 to
  26.94 on k = 10 and 8.29 to 14.21 on radius.
- **Mostly nearest-neighbour work:** one mean separation, paid for in memory,
  1,622 MiB at 1e7 against 741 at three.
- **Mostly shell or radius queries:** three to four, keeping `rmax / cellsize`
  at or below 1.
- **Memory-bound:** go coarser, but expect little further gain past three mean
  separations.

Worth more than any of the above: run the sweep on your own data, which takes a
few minutes.

```
python docs/benchmark_cellsize.py --sweep
```

## What to expect on other data

**Clustered or masked catalogues will behave differently, and the cell-side
guidance may not survive.** Every number here is uniform random points in a
cube, which is the case a uniform grid suits best. A clustered catalogue, or a
survey footprint with holes, fills some cells heavily and leaves others empty:
queries in the dense regions test far more candidates than the objects-per-cell
average suggests, while the empty cells still cost memory and still get walked.
A tree adapts its subdivision to local density and a uniform mesh does not, so
expect the comparison to move against meshsearch, and expect the best cell side
to differ from the one suggested above. Sweep it.

**Read the fine-grid memory figures as an order of magnitude.** At one mean
separation the grid holds millions of small per-cell vectors, and the resident
set depends on how the allocator grows them; it varies by hundreds of MiB for
the same data. Size a machine from the coarse-grid rows, which are stable, and
treat one mean separation at 1e7 as "one to two gigabytes" rather than a number.

**Nothing here measures threads.** Const members may be called concurrently on
one grid, and non-const members may not, but no figure above says what that
scales like.

**Nothing here measures a mixed or evolving workload.** Each table is one query
kind on a grid built once and left alone. `addObject` and `removeObject` are not
measured, nor is a grid queried after many of them. A grid that has churned
heavily is not the grid measured here: the storage a removed object held is
kept, and objects added later sit at the end of the coordinates rather than
beside the others in their cell, which is where a freshly built grid gets some
of its speed.

**Nothing here measures anisotropic boxes or non-uniform units.** The cell is
cubic and the same on all three axes. Data much longer in one dimension, or
coordinates whose axes are not comparable, gets a cell side that is a compromise
on every axis at once.
