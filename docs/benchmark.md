# Performance

Three questions: how meshsearch compares to the alternatives, how to choose the
cell side, and what none of this measures.

Every number here was produced by `docs/benchmark.py` and
`docs/benchmark_cellsize.py --sweep` on an Apple M1 Pro (10 cores, 16 GiB,
macOS 14.5), with Python 3.14.7, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1,
and meshsearch built with AppleClang at `-O3`. Nothing is tuned to favour any
library: the trees are built with default parameters and everything is queried
single-threaded.

## Against cKDTree and BallTree

Points are uniform in the unit cube from a seeded generator, with 1000 uniform
query points, `k = 10`, and a radius of twice the mean separation — which holds
the answer at about 33 neighbours at every size, so one query does a constant
amount of work as the data grows. All three libraries return the same neighbour
counts, 32.1, 32.9 and 33.2 at 1e5, 1e6 and 1e7, which is what makes the
comparison mean anything.

Query times are microseconds for one query point, the median of three repeats.
The per-call rows call the library once per point; the batched rows hand all
1000 points over in a single call. Memory is resident-set growth from holding
one structure, each library and size measured in its own process.

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

**Construction: meshsearch is faster everywhere** — about 2x against cKDTree at
one mean separation, 2.6x to 3.7x at four, where there are fewer cells to
create.

**Memory is its weakest column at a fine cell side and its strongest at a
coarse one.** At one mean separation it holds 1.5 GiB at 1e7 against cKDTree's
400 MiB, because the offset mask and the per-cell vectors both scale with the
cell count. At four mean separations the cell count falls 64-fold, the figure
drops to 672 MiB, and at 1e5 it is a third of cKDTree's.

**Nearest neighbour, one mean separation: meshsearch wins in both calling
styles.** Per call it is 5x to 8x faster than cKDTree; batched, 0.33, 0.42 and
0.58 microseconds against 0.83, 1.31 and 1.42, so 2.4x to 3.1x faster. At four
mean separations this reverses — 2.1x to 5.4x slower batched — because each
cell holds 64 candidates to test.

**Radius, four mean separations: batched meshsearch is the fastest thing
measured.** 3.09, 4.48 and 8.76 microseconds against batched cKDTree's 4.56,
6.24 and 9.22, so 1.05x to 1.48x faster at all three sizes. At one mean
separation it is 1.0x to 1.6x slower than batched cKDTree.

**k = 10 is the query to take elsewhere.** Against batched cKDTree, meshsearch
is 2.0x to 4.2x slower at one mean separation and 4.9x to 7.1x at four, and no
calling style closes it: the bounded heap pays for every candidate it rejects,
while a tree rejects them in groups.

**BallTree loses every row.** Its per-call radius query is pathological — 9
milliseconds at 1e7 against a batched 23 microseconds — which is sklearn's
per-call overhead rather than its geometry.

### Per call or batched

Subtracting the batched time from the per-call time gives what a library spends
per call before doing any work. For cKDTree that is 7.2 to 8.1 microseconds
across the three sizes; for meshsearch, **0.8 to 1.2**, because nanobind's
dispatch is cheaper than scipy's Python-level wrapper.

That sets how much batching can win. A query taking 1 microsecond gets most of
its time back; one taking 15 gets under 10 per cent. In these tables batching
gains 3.1x to 3.3x on nearest neighbour at one mean separation, and 1.1x to
1.4x on everything else. If a query is already slow, batching is not what will
fix it.

The batched forms exist in the Python bindings only. A C++ caller has no
boundary to cross and writes a loop.

## Choosing the cell side

The cell side is the one tuning parameter, and it is not a single trade-off:
the best choice for one query kind is not the best for another. Measured with
`docs/benchmark_cellsize.py --sweep`, radius held at twice the mean separation
so the answer stays at 33 neighbours throughout. Query times in microseconds,
the median of three repeats.

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

**Nearest neighbour is the one query that only gets worse.** From 1 to 4 mean
separations it goes 1.21 to 4.38 microseconds at 1e6 and 1.80 to 9.88 at 1e7, a
factor of 3.6 and 5.5. It scans a fixed neighbourhood of cells and tests
everything in it, so its cost follows the objects per cell.

**k-nearest is flat in the middle and worse at the ends,** best at 2 mean
separations at 1e6 (9.01 against 11.70 at 1 mps) and near-flat from 1 to 2 at
1e7. Past that it follows the objects per cell like nearest neighbour, reaching
a factor of 1.7 to 2.0 by 4 mps.

**A radius query is *fastest* on a coarse grid,** and its best value is in the
middle: 4.41 microseconds at 3 mean separations against 10.88 at 1, and 8.29
against 14.63 at 1e7. It sizes its scan to `Rmax` rather than scanning a fixed
neighbourhood, so a coarser grid means fewer, larger cells covering the same
sphere. The rise from 3 to 4 mps is the objects-per-cell term starting to win
again.

**Memory falls, but nothing like the cell count does.** At 1e7 it goes 1,622 to
701 MiB, a factor of 2.3 for a 64-fold drop in cells. Two structures scale with
the cell count and account for the saving: the offset mask, 8 entries of 12
bytes per cell, is about 923 MiB at 1 mps against 14 at 4; and the per-cell
vector headers, 24 bytes each, another 230 MiB against 4. Underneath them is a
floor the cell side cannot touch — the grid's own copy of the coordinates
(229 MiB at 1e7), the two index tables it keeps (76 MiB), and the bucket
contents (38 MiB) — which is why 3 mps and 4 mps barely differ.

**Cost grows with n at a coarse cell side, even though the work does not.** At
a fixed cell side in mean separations, the objects per cell and the cells
scanned are both constant, so a query performs the same number of distance
tests at 1e7 as at 1e6. It nonetheless costs 2.26x on nearest neighbour from
1e6 to 1e7 at 4 mps, against 1.49x at 1 mps. What grows is the cost of reaching
memory, not the amount of geometry.

### How far a shell query reaches

A shell query scans whole layers of cells around the query point, where the
layer index is a lower bound on distance: a layer-`L` cell is at least `L` cell
sides away. The scan therefore covers layers 0 to `floor(Rmax / cellsize)`,
which makes `Rmax / cellsize` — not `Rmax`, and not the cell side alone — the
number that decides the work:

| Rmax / cellsize | layers scanned | cells visited | of those, can hold a member |
|---|---|---|---|
| 0.25 | 0 | 27 | 3.0 |
| 0.50 | 0 | 27 | 6.9 |
| 1.00 | 0–1 | 125 | 20.7 |
| 2.00 | 0–2 | 311 | 84.2 |
| 4.00 | 0–4 | 1,015 | 443.9 |

This is geometry, not a timing: the cell counts follow from the layer rule, and
the last column is a 400-sample average over where the query point sits inside
its own cell. Multiply cells visited by objects per cell for the number of
distance tests a shell query performs.

The cells that are scanned but cannot hold a member are inherent to grouping
offsets by the floor of their minimum distance: a layer admitted because part
of it is in range brings the rest of itself along.

### Starting point

- **Mixed use:** two to three mean separations. Nothing is at its worst there,
  and at 1e7 it costs 3.24 to 6.72 microseconds on nearest neighbour, 15.75 to
  26.94 on k = 10 and 8.29 to 14.21 on radius.
- **Mostly nearest-neighbour work:** one mean separation, and pay for it in
  memory — 1,622 MiB at 1e7 against 741 at three.
- **Mostly shell or radius queries:** three to four, keeping `Rmax / cellsize`
  at or below 1 so the scan stays within layers 0 and 1.
- **Memory-bound:** go coarser, but expect little further gain past three mean
  separations.

The sweep is cheap to repeat on real data, which is worth more than any of the
above: `python docs/benchmark_cellsize.py --sweep`.

## What was not measured

**Clustered data.** Uniform random points in a cube are the case a uniform grid
suits best. A clustered catalogue, a survey footprint with holes, or a strongly
anisotropic box would change the picture, probably against meshsearch, because
its cell side is global while a tree adapts to local density. Nothing here
measures that, and the cell-side guidance above may not survive it.

**Anything but one machine, one compiler and one Python.** Times are medians of
three repeats, enough to rank effects of this size but not to separate
differences under about 10 per cent. Treat a 1.05x as a tie.

**Threads.** Every measurement is single-threaded. Const members may be called
concurrently on one grid, but no scaling across threads is measured here.

**Memory precisely.** The figure is a resident-set high-water mark, and it is
the noisiest number in this file: 1e7 at one mean separation reads 1,495 MiB in
one table above and 1,622 in another, the same grid measured in two processes,
because it depends on how the allocator grows ten million small vectors. The
coarse-grid rows are stable to a few MiB. Read the fine-grid memory figures as
an order of magnitude.

**Mixed workloads, removals and additions.** Each table measures one query kind
on a static grid. `addObject` and `removeObject` are not timed, and neither is
a grid that has been heavily modified after construction.
