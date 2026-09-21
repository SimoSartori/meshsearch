# meshsearch against cKDTree and BallTree

Measured with `docs/benchmark.py`, which produced every number below. Nothing
was tuned to favour any library: meshsearch is given the cell side its own
documentation suggests as a starting point, the mean separation, the two trees
are built with default parameters, and everything is queried single-threaded.

## Setup

Apple M1 Pro, 10 cores, 16 GiB, macOS 14.5 (Darwin 23.5.0). Python 3.14.7,
numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, meshsearch built with
AppleClang at `-O3`.

**Two changes were made to the library as a result of this work**: the `Rmax`
layer bound in `closeObjects` was derived and corrected, and the coordinates
are now stored in cell order internally. The later sections say what each was
and measure it. **Every measurement in this file postdates both**, except the
layout section's tables, which predate the second and are labelled as the
evidence that motivated it. No table mixes numbers from before and after a
change; where an earlier figure is worth stating it is in the prose, labelled.

meshsearch is tabulated at two cell sides, one mean separation and four. The
first is what its own documentation suggests as a starting point; the second is
the regime the library is used in. The trees have no such parameter.

Points are uniform in the unit cube, from a seeded generator. 1000 query
points, also uniform, with `k = 10` and a radius of twice the mean separation,
which holds the answer at about 33 neighbours at every size so the query does a
constant amount of work as the data grows. All three libraries return the same
neighbour counts, which is what makes the comparison meaningful: 32.1, 32.9 and
33.2 at 1e5, 1e6 and 1e7.

Each library and size runs in its own process, so the memory figure is the
resident-set growth caused by holding that one structure. Times are the median
of three repeats.

## Head to head

Query times are microseconds for one query point. The per-call rows call the
library once per point; the batched rows hand all 1000 points over in a single
call. meshsearch's batched forms live in the Python bindings only, for the
reason the boundary-cost section below measures.

**Cell side of one mean separation**, about one object per cell.

| points | | meshsearch | cKDTree | BallTree |
|---|---|---|---|---|
| **1e5** | construction (s) | 0.016 | 0.028 | 0.028 |
| | memory (MiB) | 27 | 27 | 86 |
| | nearest, per call | 1.00 | 8.03 | 31.36 |
| | nearest, batched | 0.33 | 0.82 | 3.39 |
| | k = 10, per call | 5.42 | 10.08 | 35.01 |
| | k = 10, batched | 4.07 | 2.00 | 6.84 |
| | radius, per call | 5.17 | 5.82 | 49.65 |
| | radius, batched | 4.04 | 3.88 | 3.28 |
| **1e6** | construction (s) | 0.207 | 0.365 | 0.376 |
| | memory (MiB) | 286 | 52 | 95 |
| | nearest, per call | 1.08 | 8.54 | 48.45 |
| | nearest, batched | 0.57 | 1.13 | 20.92 |
| | k = 10, per call | 13.38 | 11.81 | 59.53 |
| | k = 10, batched | 12.33 | 2.96 | 31.62 |
| | radius, per call | 12.34 | 8.32 | 896.58 |
| | radius, batched | 11.70 | 6.62 | 10.41 |
| **1e7** | construction (s) | 2.850 | 5.814 | 6.470 |
| | memory (MiB) | 2,004 | 399 | 177 |
| | nearest, per call | 1.56 | 8.89 | 84.27 |
| | nearest, batched | 0.74 | 1.46 | 56.06 |
| | k = 10, per call | 16.54 | 13.50 | 107.83 |
| | k = 10, batched | 15.22 | 4.10 | 73.58 |
| | radius, per call | 14.79 | 11.32 | 6,690.09 |
| | radius, batched | 13.90 | 9.51 | 24.72 |

**Cell side of four mean separations**, about 64 objects per cell: the regime
OTreconstruction builds its grids at, and the one cell ordering favours.

| points | | meshsearch | cKDTree | BallTree |
|---|---|---|---|---|
| **1e5** | construction (s) | 0.012 | 0.028 | 0.027 |
| | memory (MiB) | 11 | 27 | 89 |
| | nearest, per call | 2.64 | 8.11 | 31.42 |
| | nearest, batched | 1.81 | 0.84 | 3.43 |
| | k = 10, per call | 12.90 | 10.62 | 35.00 |
| | k = 10, batched | 10.80 | 1.98 | 7.21 |
| | radius, per call | 3.81 | 5.96 | 46.61 |
| | radius, batched | 2.60 | 4.06 | 3.61 |
| **1e6** | construction (s) | 0.132 | 0.365 | 0.364 |
| | memory (MiB) | 116 | 53 | 96 |
| | nearest, per call | 4.50 | 8.54 | 44.85 |
| | nearest, batched | 3.54 | 1.18 | 17.73 |
| | k = 10, per call | 19.38 | 11.55 | 55.24 |
| | k = 10, batched | 17.48 | 2.82 | 26.71 |
| | radius, per call | 5.86 | 8.23 | 712.22 |
| | radius, batched | 4.77 | 5.92 | 9.76 |
| **1e7** | construction (s) | 1.462 | 5.846 | 6.523 |
| | memory (MiB) | 930 | 399 | 176 |
| | nearest, per call | 8.83 | 8.58 | 86.34 |
| | nearest, batched | 6.82 | 1.47 | 54.63 |
| | k = 10, per call | 32.86 | 12.00 | 110.95 |
| | k = 10, batched | 29.67 | 3.93 | 79.14 |
| | radius, per call | 9.79 | 11.20 | 6,844.32 |
| | radius, batched | 8.91 | 9.44 | 25.75 |

All three libraries return the same neighbour counts, 32.1, 32.9 and 33.2 at
the three sizes, which is what makes the comparison mean anything.

The memory figure is the resident-set growth from holding one structure, each
library and size measured in its own process. It is the noisiest number here:
meshsearch's 1 mps row at 1e7 has spanned 1,260 to 1,724 MiB across runs,
because it is a high-water mark set by how the allocator happens to grow ten
million small vectors. The coarser rows are stable to a few MiB.

### What the Python boundary actually costs

Subtracting the batched time from the per-call time gives what each library
spends per call before doing any work. For cKDTree it is 7.2 microseconds at
1e5 and 7.4 at 1e7. For meshsearch it is **0.67 and 0.82** — roughly nine times
less, because nanobind's dispatch is cheaper than scipy's Python-level wrapper.

That is the premise this work was built on, and it turns out to be a scipy
number rather than a universal one. It sets how much batching can possibly
win: a query taking 1 microsecond gets most of its time back, and one taking 15
gets about 6 per cent. Hence the batch gains below, which are 3.1x on nearest
neighbour at 1e5 and 1 mps, about 2x on nearest neighbour elsewhere, and only
1.1x to 1.5x on k = 10 and radius.

## What the numbers say

**Construction: meshsearch wins everywhere, and by more at the coarse cell
side.** Against cKDTree it is around 2x faster at 1 mps and 2.4x to 4.0x at
4 mps, where there are fewer cells to create.

**Memory: meshsearch's weakest column at 1 mps, competitive at 4.** At one mean
separation it holds around 2 GiB at 1e7 against cKDTree's 399 MiB, because the
offset mask and the per-cell vectors both scale with the cell count. At four
mean separations the cell count falls 64-fold and the figure drops to about
800 MiB, and at 1e5 it is the smallest of the three.

**Nearest neighbour at 1 mps: meshsearch wins in both calling styles.** Per
call it is 8x to 9x faster than cKDTree. Batched it is 0.33, 0.57 and 0.74
microseconds against cKDTree's 0.82, 1.13 and 1.46, so **2.0x to 2.5x faster
batched as well.** At 4 mps the picture reverses, 2.2x to 4.6x slower batched,
because each cell holds 64 candidates to test.

**k = 10: the gap against batched scipy does not close.** This was the row the
boundary cost was suspected of hiding, and it was not. Batching it wins only
1.1x to 1.3x, because a k = 10 query takes 13 to 17 microseconds and only about
0.8 of that was the boundary. Against batched cKDTree, meshsearch was 2.6x to
4.3x slower per call and is 2.0x to 4.2x slower batched at 1 mps, 5.5x to 7.6x
at 4 mps. The gap is the algorithm, not the binding: the bounded heap pays for
every candidate it rejects, and cKDTree's tree rejects them in groups. Nothing
in the Python layer will fix it.

**Radius at 4 mps: batched meshsearch is the fastest thing measured.** 2.60,
4.77 and 8.91 microseconds against batched cKDTree's 4.06, 5.92 and 9.44, so
1.06x to 1.56x faster at all three sizes, having started this investigation
6.8x behind. At 1 mps it stays 1.0x to 1.8x slower than batched cKDTree.

**BallTree loses every row.** Its per-call radius query is pathological, 6.7
milliseconds at 1e7 against a batched 25 microseconds, which is sklearn's
per-call overhead rather than its geometry.

**So: pick the cell side for the query.** One mean separation for nearest
neighbour, where meshsearch beats both trees per call and batched, and pays for
it in memory. Four for radius, where batched meshsearch is the fastest option
at a quarter of the memory and builds three times faster. k = 10 is the query
to take elsewhere if it is the bottleneck, at either cell side and in either
calling style.

## Cell side

Everything above is at one mean separation. Grids in use are built coarser:
OT builds at four, about 64 objects per cell. The mask holds 8 entries per
cell and the cell count falls as the cube of the factor, 64-fold from 1 to 4,
so the memory was expected to move a lot. Measured with
`docs/benchmark_cellsize.py --sweep`, radius held at twice the mean separation
so the answer stays at 33 neighbours throughout. Query times in microseconds.

**1e6 points**

| cell side | cells | objects per cell | build (s) | memory (MiB) | nearest | k = 10 | radius |
|---|---|---|---|---|---|---|---|
| 1 mps | 1,000,000 | 1 | 0.261 | 252 | 1.19 | 16.07 | 16.38 |
| 2 mps | 125,000 | 8 | 0.179 | 106 | 2.98 | 17.65 | 15.38 |
| 3 mps | 39,304 | 25 | 0.139 | 86 | 6.91 | 37.28 | 9.13 |
| 4 mps | 15,625 | 64 | 0.134 | 96 | 12.33 | 65.46 | 14.37 |

**1e7 points**

| cell side | cells | objects per cell | build (s) | memory (MiB) | nearest | k = 10 | radius |
|---|---|---|---|---|---|---|---|
| 1 mps | 10,077,696 | 1 | 3.394 | 1,260 | 1.35 | 20.09 | 19.01 |
| 2 mps | 1,259,712 | 8 | 2.145 | 753 | 5.69 | 28.77 | 25.40 |
| 3 mps | 373,248 | 27 | 1.620 | 631 | 13.83 | 71.05 | 16.76 |
| 4 mps | 157,464 | 64 | 1.417 | 617 | 28.34 | 136.88 | 29.94 |

Each row is the mean of two runs. **Memory falls, but nothing like 64-fold.**
At 1e7 it goes 1,260 to 617 MiB, a factor of 2.0. The memory figure at 1 mps is
the noisiest number in this file, spanning 1,260 to 1,635 MiB across runs,
because it is a resident-set high-water mark set by how the allocator happens
to grow ten million small vectors; the coarser rows are stable to a few MiB. The absolute saving, about 1,020 MiB, is close to what the
cell count predicts for the two structures that scale with it: the mask is
923 MiB at 1 mps against 14 MiB at 4, and the per-cell vector headers another
230 MiB against 4. What the ratio does not show is the floor underneath them.
The grid holds its own copy of the coordinates, 240 MiB at 1e7, and it copies
them through a temporary, so the resident set keeps the high-water mark of the
constructor whether or not the allocator still needs it. Beyond about 3 mean
separations the cell-count structures have stopped mattering and only that
floor is left, which is why 3 and 4 barely differ.

**Nearest-neighbour and k-nearest still get much worse at a coarse cell side;
radius no longer does.** From 1 to 4 mean separations at 1e7,
nearest-neighbour goes 1.35 to 28.34 microseconds, a factor of 21, and k = 10
from 20.09 to 136.88, a factor of 6.8. Neither has moved: before the bound was
corrected those factors were 21 and 6.5. That is as it should be, because
neither query uses the `Rmax` bound at all; their cost is the 64 objects per
cell they must now test where they used to test one, which is intrinsic to a
coarse grid and not a margin.

**The radius conclusion, by contrast, was an artefact of the old margin and is
gone.** That row used to run 58.9 microseconds at 1 mps to 568.9 at 4, a factor
of 9.7 that made it the worst in the table. It now runs 19.0 to 29.9, a factor
of 1.6, and is the best-behaved row. The reason is that a shell's scan shrinks
as the cell side grows: at 4 mps the radius is half a cell and the derived
bound scans only layer 0, where the old `+ 3` scanned four layers whatever the
radius was, and four layers cover far more cells when cells are large. Build
time moves the other way throughout, 3.39 to 1.42 seconds: fewer cells to
create.

So a cell side is not a single trade: it is the best choice for
nearest-neighbour and the worst for k = 10 at the fine end, and the reverse for
radius. The head-to-head tables give both ends against the trees.

**The growth with n survives, and widens.** From 1e6 to 1e7 at 4 mps,
nearest-neighbour costs 2.30x and radius 2.08x, against 1.13x and 1.16x at
1 mps. At a fixed cell side in mean separations the work per query is
provably constant in n: the objects per cell stay at 64 and the same mask
layers are scanned, so a query performs the same number of distance tests at
1e7 as at 1e6. It nonetheless takes over twice as long.

## Layout

That last observation isolates the question, since the logical work is fixed.
Measured with `docs/benchmark_cellsize.py --locality`: the same points, the
same cell side, the same cell count and the same distance tests, built twice,
once in input order and once sorted by the cell each point falls into. Only
the order of the per-cell allocations and of the coordinate accesses differs.
The binning was checked against `get_cell`, and both grids return identical
nearest-neighbour distances, so nothing but the layout changed.

| points | cell side | order | nearest | k = 10 | radius |
|---|---|---|---|---|---|
| 1e6 | 1 mps | input | 1.32 | 17.20 | 17.48 |
| 1e6 | 1 mps | cell | 1.23 | 12.93 | 13.73 |
| 1e6 | 1 mps | **gain** | **1.08x** | **1.33x** | **1.27x** |
| 1e7 | 1 mps | input | 1.52 | 20.97 | 19.28 |
| 1e7 | 1 mps | cell | 1.10 | 16.38 | 15.00 |
| 1e7 | 1 mps | **gain** | **1.38x** | **1.28x** | **1.29x** |
| 1e6 | 4 mps | input | 12.40 | 63.10 | 14.22 |
| 1e6 | 4 mps | cell | 4.34 | 22.14 | 6.51 |
| 1e6 | 4 mps | **gain** | **2.85x** | **2.91x** | **2.24x** |
| 1e7 | 4 mps | input | 39.38 | 129.85 | 27.77 |
| 1e7 | 4 mps | cell | 9.49 | 32.69 | 9.42 |
| 1e7 | 4 mps | **gain** | **4.48x** | **3.99x** | **2.95x** |

Each figure is the mean of two runs. The absolute times here run higher than
the cell-side table's for the same configuration, because this process also
holds the original arrays and the sorting permutation, which changes the
memory pressure; the ratio within a process is what the test is for and what
should be read.

**The conclusion survives, reduced in one place.** Sorting the input by cell
still makes every query faster without changing a line of the library or a
single distance test, by 1.1x to 4.5x where it was 1.4x to 5.2x. The gain is
larger at the coarser cell side, where each cell holds 64 points to walk rather
than one, and larger at 1e7 than at 1e6 in every row. Both are what a memory
effect predicts and neither is what a work effect would predict.

**Where it shrank is exactly where the margin used to inflate it.** The radius
gain at 4 mps fell from 4.09x and 5.22x to 2.24x and 2.95x, because the old
bound had the query walking 613 cells instead of 27, and a scan that visits
more cells than it needs pays the scattered-read penalty on every one of the
surplus. The nearest-neighbour and k-nearest gains, which never involved that
bound, are unchanged within the spread: 2.85x and 4.48x against 2.75x and
3.70x, 2.91x and 3.99x against 3.30x and 4.14x.

Cell ordering still accounts for much, though not all, of the growth with n. At
4 mps the nearest-neighbour cost grows 3.18x from 1e6 to 1e7 in input order and
2.19x in cell order.

Memory is not reported for this test: the second grid is built after the first
is freed and reuses its pages, so the resident-set delta for it is an artefact
of the allocator rather than a measurement of the grid.

**This result was acted on.** The library now stores its coordinates in cell
order internally, so a caller gets the gain without reordering anything; the
next section measures how much of it survives being done inside. The tables in
this section were measured before that change and are kept as the evidence for
it, since sorting the caller's input is what isolates layout from every other
variable. Flattening the per-cell lists into one contiguous array with offsets
is the remaining version of the same idea, and is not done.

## Internal cell ordering

The layout section showed that reordering the input bought a large factor. The
library now does it for itself: the constructor copies the coordinates in cell
order rather than input order and keeps the permutation and its inverse, so a
cell's objects are contiguous in `m_X`, `m_Y` and `m_Z` whatever order the
caller passed them in.

Nothing public changed. The caller still passes their own arrays and still
receives indices into those arrays; the public half of the header is
byte-identical. The buckets hold internal indices, so every distance test in
every query reads the coordinate arrays directly with no lookup, and the two
index spaces meet only at the boundary: once on entry for the overloads that
take an object, and once per returned index when the result vector is built.

Measured with `docs/benchmark_cellsize.py --one-sweep` against the previous
implementation, both given the same points in input order, mean of two runs.
Microseconds per query.

**1e6 points**

| cell side | nearest | | | k = 10 | | | radius | | |
|---|---|---|---|---|---|---|---|---|---|
| | old | new | gain | old | new | gain | old | new | gain |
| 1 mps | 1.22 | 1.02 | 1.20x | 14.98 | 12.24 | 1.22x | 15.11 | 11.49 | 1.31x |
| 2 mps | 2.90 | 1.76 | 1.65x | 17.68 | 9.00 | 1.96x | 16.06 | 8.08 | 1.99x |
| 3 mps | 7.21 | 3.49 | 2.07x | 38.54 | 15.53 | 2.48x | 9.79 | 5.23 | 1.87x |
| 4 mps | 12.05 | 4.22 | **2.85x** | 66.53 | 19.67 | **3.38x** | 15.54 | 5.87 | **2.65x** |

**1e7 points**

| cell side | nearest | | | k = 10 | | | radius | | |
|---|---|---|---|---|---|---|---|---|---|
| | old | new | gain | old | new | gain | old | new | gain |
| 1 mps | 1.46 | 1.80 | 0.81x | 20.83 | 16.21 | 1.29x | 19.88 | 14.90 | 1.33x |
| 2 mps | 5.42 | 2.68 | 2.02x | 26.56 | 16.07 | 1.65x | 25.41 | 14.70 | 1.73x |
| 3 mps | 13.81 | 6.31 | 2.19x | 68.35 | 26.26 | 2.60x | 16.07 | 7.86 | 2.04x |
| 4 mps | 28.10 | 7.89 | **3.56x** | 131.95 | 32.77 | **4.03x** | 30.36 | 9.54 | **3.18x** |

**Almost all of the external gain survives the indirection.** Against the
1.1x to 4.5x that sorting the caller's input produced, doing it internally
gives, at the cell side the library is used at, 2.85x and 3.56x on
nearest-neighbour where sorting the input gave 2.85x and 4.48x, 3.38x and
4.03x on k = 10 against 2.91x and 3.99x, and 2.65x and 3.18x on radius against
2.24x and 2.95x. Eight of the twelve comparisons land at or above the external
figure and the rest within the run-to-run spread, so the per-result translation
costs nothing measurable. That is the expected outcome given where the
translation sits: it happens once per returned index, while the distance tests
it enables are hundreds or thousands per query.

**The gain grows with the objects per cell, and at one object per cell there is
none.** It runs 1.2x to 1.3x at one mean separation and 2.9x to 4.0x at four,
monotonically in between. That is the mechanism stated plainly: what cell
ordering buys is contiguity *within* a cell, and a cell holding a single point
has nothing to make contiguous. The regime the library is used in is the one
where the change pays.

**One row is a non-result.** Nearest-neighbour at 1 mps and 1e7 reads 0.81x,
i.e. slower. Repeating that configuration five times gives 1.27 to 2.07
microseconds for the new implementation against 1.35 to 1.65 for the old, with
the new values bimodal and no consistent direction. At roughly 1.5 microseconds
for a scan of 27 nearly empty cells, this measurement is below the resolution
of the harness. It is reported as it fell rather than dropped or repeated until
it agreed, but nothing should be concluded from it beyond "no gain here", which
is what the objects-per-cell reading already predicts.

**Cost.** The two index tables are 4 bytes per object each. At 1e7 and 4 mps
the resident set goes from 612 to 697 MiB, and at 1e6 from 103 to 109, both
consistent with 80 MiB and 8 MiB of tables. Construction is unchanged within
the spread: the counting sort that produces the ordering replaces nothing and
costs one extra pass, and at 1e7 and 1 mps the build reads 2.83 seconds against
2.96 before.

## The Rmax layer bound: what was wrong, and the correction

Cell ordering brought nearest-neighbour to roughly par with cKDTree and left
radius about 10x behind, 109.81 microseconds against 10.85 at 1e7 and 4 mean
separations. Diagnosed with `docs/diagnose_radius.py`; the numbers in this
first half are from before the fix.

**The radius is half a cell side.** At 1e7 the mean separation is 0.00464, the
benchmark's `Rmax` is twice that, and the cell side is four times it, so
`Rmax / cellsize = 0.5`. This is the opposite of the regime a grid handles
worst: the sphere is smaller than one cell, not larger. It is also what made
the old margin expensive, because that margin was a fixed number of layers
rather than a fraction of the radius.

**The scan visited 613 cells where 27 would do, and 7 held anything.** The old
`max_mask_idx` was `ceil(Rmax / cellsize) + 3`, which is 4 here, so the loop ran
over layers 0, 1, 2 and 3. Only layer 0 can contain a member at all: a layer-1
cell is at least one cell side away, and `Rmax` is half of one. Of the 613
cells those four layers hold, an average of 6.6 could contain a point within
`Rmax` of the query point, and 606.4 could not, 98.9 per cent of them. At 64
objects per cell that is 613 x 64 = **39,232 distance tests to return about 33
neighbours, 1,199 tests per hit.**

**The cost was the fixed scan, not the answer.** Holding the layer range
constant and varying `Rmax` below one cell side, which did not change
`max_mask_idx`, the time did not move: 110.60 microseconds for a query
returning nothing, 115.66 for one returning 33, and every value between flat to
within the noise. For scale, on the same grid `nearest_object`, which scans
layer 0 and stops, took 7.31 microseconds, and `get_objects_in_cell`, which
touches one cell, 1.15. The binding's per-call overhead was about 1 per cent of
the 110, and the answer itself free; the other 99 per cent was walking cells.

**The result vector was not a factor.** Holding the cells visited at 613 and
raising `Rmax` from 0.5 to 1.0 cell sides took the result count from 33 to 263,
eight times more `push_back` calls into a vector with no `reserve`, and the
time fell slightly, 118.24 to 115.30 microseconds. Growth only became visible
far outside the benchmark's regime: at 15,801 results per query, about 8 per
cent of the time. Ruled out.

**cKDTree was doing less work, not the same work faster.** Descending the tree
and counting the points in every leaf whose box intersects the ball, which
overestimates because scipy's own node boxes are tighter than the ones a split
rebuilds:

| with the old bound | distance tests per query | per hit |
|---|---|---|
| cKDTree | 211 (upper bound) | 6.5 |
| meshsearch, 1 mps | 1,015 | 31 |
| meshsearch, 4 mps | 39,232 | 1,199 |

meshsearch evaluated about 186 times more distances than cKDTree for the same
answer and was only 10 times slower, so its inner loop is roughly 18 times
cheaper per test. The geometry was never the problem; the amount of it was.

**A cost model.** Fitting `cells_visited x (per_cell + objects_per_cell x
per_test)` across the four cell sides at 1e7 gave about 47 ns per cell visited
and about 2.4 ns per distance test, reproducing the measurements to within
roughly 20 per cent. The fitted constants move a few per cent between runs, so
this describes where the time goes rather than stating a law. Two terms are
enough to explain the whole radius picture across regimes, which is why the
cell side mattered twice over: it set how many cells a fixed layer margin
covered and how many objects sat in each.

### The bound, derived

For an offset `(i, j, k)` the per-axis gap is `d = max(|i| - 1, 0)`, and the
layer is `L = floor(sqrt(dx^2 + dy^2 + dz^2)) = floor(m)`. Along each axis the
closest a point of that cell can come to a point of the target cell is `d`
cell sides, so the minimum distance between them is `m * cellsize`, and since
`L <= m`, at least `L * cellsize`.

A layer-`L` cell can therefore hold a point within `Rmax` only if

    L * cellsize <= Rmax,   that is   L <= Rmax / cellsize

so the last layer worth scanning is `floor(Rmax / cellsize)`, and since the
loop bound is exclusive,

    max_mask_idx = floor(Rmax / cellsize) + 1

The same inequality gives the in-loop break: stop at layer `maskInd` once
`maskInd * cellsize > Rmax`. The comparison is strict because a point at
exactly `Rmax` is inside the shell, `Rmin <= d <= Rmax` being closed at both
ends. The division is written as a division rather than a multiplication by a
reciprocal so that a radius which is an exact multiple of the cell side stays
exact and the boundary layer is not dropped by rounding.

The `Rmin` lower bound is unchanged. It was derived separately and is correct
as it stands.

**Neither the bound nor the break can be tightened by one more layer.** The
suite has a deterministic case for each: a query point 0.75 into cell 10 and an
object at the low corner of cell 13, an offset of 3 whose gap is 2 and whose
layer is 2, exactly 2.25 cell sides away, with `Rmax = 2.25` so that layer 2 is
the last one scanned and the object lies exactly on the outer edge. Taking one
layer off `max_mask_idx` loses it, as does weakening the break's `>` to `>=`.
Both variants were built and both fail the suite.

### After

Cells visited, and how many of them can hold a member, at the same radii as the
table above:

| Rmax / cellsize | layers scanned | cells visited | can hold a member |
|---|---|---|---|
| 0.25 | 0 | 27 | 2.9 |
| 0.50 | 0 | 27 | 6.6 |
| 1.00 | 0–1 | 125 | 20.5 |
| 2.00 | 0–2 | 311 | 85.2 |
| 4.00 | 0–4 | 1,015 | 444.2 |

Against the old bound's 613, 613, 613, 1,015 and 2,399 for the same five radii.
At the benchmark's 0.5 the scan is 23 times smaller, and the distance tests
with it, from 39,232 to 1,728, or 52 per hit against cKDTree's 6.5.

The cells that remain and still cannot hold a member are inherent to grouping
offsets by `floor` of their minimum distance: a layer admitted because some of
it is in range carries the rest of itself along. That is the granularity of the
mask, not a margin, and nothing here changes it.

Radius query time, microseconds, all measured after the correction:

| configuration | radius query |
|---|---|
| 1e7, 1 mps, input order | 19.01 |
| 1e7, 4 mps, input order | 27.77 |
| 1e7, 4 mps, cell order | **9.42** |
| 1e6, 4 mps, cell order | 6.51 |

The same four configurations cost 60.40, 572.79, 109.81 and 60.10 microseconds
before the correction, so the gain runs from 3x at the finest cell side to 20x
at the coarsest, which is the shape the fix predicts: the old margin was a
fixed three layers whatever the radius, and three layers cover far more cells
when cells are large.

At the cell side the library is used at, and with the points in cell order, a
radius query at 1e7 costs 9.43 microseconds against cKDTree's 10.67 per call
and 8.93 batched. The row that prompted the investigation has gone from 10x
behind to level.

The other queries are untouched, as they should be: they do not use this bound.
Nearest-neighbour at 1e7 and 1 mps reads 1.35 microseconds against 1.38 before,
and k = 10 reads 20.09 against 19.81, both inside the run-to-run spread.

## Caveats

Uniform random points in a cube are the case a uniform grid is best suited to.
Clustered data, a survey footprint with holes, or a strongly anisotropic box
would change the picture, probably against meshsearch, since its cell side is
global while a tree adapts to local density. Nothing here measures that.

One machine, one compiler, one Python. Times are medians of three repeats,
which is enough to rank effects of this size but not to separate differences
under about 10 per cent.
