"""Tests for the Python bindings.

They cover the guarantees the C++ suite covers -- checked the same way, against
brute force over random points -- plus what only exists at the language
boundary: the returned dtypes, the properties and len(), and the mapping of the
two C++ exception types onto Python ones.
"""

import copy
import os
import subprocess
import sys
import sysconfig
import threading

import numpy as np
import pytest

import meshsearch


# --------------------------------------------------------------------------
# helpers


def cloud(n, seed=0):
    """n points spread uniformly through the unit box."""
    rng = np.random.default_rng(seed)
    return rng.random(n), rng.random(n), rng.random(n)


def distances(x, y, z, X, Y, Z):
    return np.sqrt((X - x) ** 2 + (Y - y) ** 2 + (Z - z) ** 2)


def brute_nearest(x, y, z, X, Y, Z, alive, skip=None):
    d = distances(x, y, z, X, Y, Z)
    mask = alive.copy()
    if skip is not None:
        mask[skip] = False
    return d[mask].min()


def brute_ranked(x, y, z, X, Y, Z, alive, skip=None):
    d = distances(x, y, z, X, Y, Z)
    mask = alive.copy()
    if skip is not None:
        mask[skip] = False
    return np.sort(d[mask])


def brute_order(x, y, z, X, Y, Z, alive, skip=None):
    """Live indices ordered by (squared distance, index). The squared distance
    is computed as the library computes it, so equal distances are equal
    here too, and sorting on it rather than on its root keeps ties exact."""
    d2 = (X - x) ** 2 + (Y - y) ** 2 + (Z - z) ** 2
    mask = alive.copy()
    if skip is not None:
        mask[skip] = False
    index = np.flatnonzero(mask)
    return index[np.lexsort((index, d2[index]))]


def brute_shell(x, y, z, X, Y, Z, alive, rmax, rmin, skip=None):
    d = distances(x, y, z, X, Y, Z)
    mask = alive & (d <= rmax) & (d >= rmin)
    if skip is not None:
        mask[skip] = False
    return set(np.flatnonzero(mask).tolist())


BOX = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

# the mean separation of 2000 points in the unit box, and cell sizes well above
# and well below it, so the queries are exercised in both regimes
MPS = 2000 ** (-1 / 3)
REGIMES = [("dense", 4 * MPS), ("matched", MPS), ("sparse", 0.3 * MPS)]


@pytest.fixture(scope="module")
def points():
    return cloud(2000, seed=20260921)


@pytest.fixture(scope="module")
def grids(points):
    X, Y, Z = points
    return {name: meshsearch.MeshGrid(X, Y, Z, size, limits=BOX) for name, size in REGIMES}


# --------------------------------------------------------------------------
# the boundary itself


def test_module_surface():
    assert sorted(n for n in dir(meshsearch) if not n.startswith("_")) == [
        "Batch",
        "Error",
        "IndexError",
        "MeshGrid",
    ]


def test_method_names_are_snake_case_with_no_aliases():
    """One name per language: the C++ spellings are not also exposed."""
    surface = {n for n in dir(meshsearch.MeshGrid) if not n.startswith("_")}
    assert surface == {
        "add_object",
        "cellsize",
        "close_objects",
        "contains",
        "count_objects",
        "get_cell",
        "get_cell_coords",
        "get_objects_in_cell",
        "is_alive",
        "lims",
        "n_cells",
        "nearest_object",
        "nearest_objects",
        "periodic",
        "remove_object",
    }
    for absent in ("nearestObject", "closerObject", "get_lims", "get_nCells",
                   "get_cellsize", "get_nObjects", "n_objects", "isAlive",
                   "removeObject", "addObject"):
        assert not hasattr(meshsearch.MeshGrid, absent)


def test_len_is_the_live_count(points):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, MPS, limits=BOX)
    assert len(grid) == len(X)
    grid.remove_object(0)
    assert len(grid) == len(X) - 1
    grid.add_object(0.5, 0.5, 0.5)
    assert len(grid) == len(X)


def test_properties(points):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, 0.25, limits=BOX)

    assert isinstance(grid.cellsize, float)
    assert grid.cellsize == 0.25

    assert isinstance(grid.n_cells, tuple)
    assert grid.n_cells == (4, 4, 4)
    assert all(isinstance(c, int) for c in grid.n_cells)

    assert isinstance(grid.lims, np.ndarray)
    assert grid.lims.dtype == np.float64
    assert grid.lims.shape == (3, 2)
    assert np.allclose(grid.lims, BOX)


def test_properties_are_read_only(points):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, 0.25, limits=BOX)
    for name in ("cellsize", "n_cells", "lims"):
        with pytest.raises(AttributeError):
            setattr(grid, name, 1)


def test_index_results_are_uint32_arrays(grids):
    grid = grids["matched"]
    for result in (
        grid.nearest_objects(5, 0.5, 0.5, 0.5),
        grid.nearest_objects(5, 0),
        grid.close_objects(0.5, 0.5, 0.5, 0.2),
        grid.close_objects(0, 0.2),
        grid.get_objects_in_cell(0.5, 0.5, 0.5),
    ):
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint32
        assert result.ndim == 1


def test_scalar_results_are_plain_ints(grids):
    grid = grids["matched"]
    assert isinstance(grid.nearest_object(0.5, 0.5, 0.5), int)
    assert isinstance(grid.get_cell(0.5, 0.5, 0.5), int)
    coords = grid.get_cell_coords(0.5, 0.5, 0.5)
    assert isinstance(coords, tuple) and len(coords) == 3


def test_empty_result_is_still_a_uint32_array(grids):
    empty = grids["matched"].nearest_objects(0, 0.5, 0.5, 0.5)
    assert empty.dtype == np.uint32
    assert empty.shape == (0,)


def test_accepts_lists_and_numpy_arrays(points):
    X, Y, Z = points
    from_arrays = meshsearch.MeshGrid(X, Y, Z, MPS, limits=BOX)
    from_lists = meshsearch.MeshGrid(X.tolist(), Y.tolist(), Z.tolist(), MPS, limits=BOX)
    assert len(from_arrays) == len(from_lists)
    assert from_arrays.nearest_object(0.5, 0.5, 0.5) == from_lists.nearest_object(0.5, 0.5, 0.5)


def test_mismatched_input_lengths_raise(points):
    X, Y, Z = points
    with pytest.raises(meshsearch.Error):
        meshsearch.MeshGrid(X[:-1], Y, Z, MPS, limits=BOX)
    with pytest.raises(meshsearch.Error):
        meshsearch.MeshGrid(X, Y[:-1], Z, MPS, limits=BOX)
    with pytest.raises(meshsearch.Error):
        meshsearch.MeshGrid(X, Y, Z[:-1], MPS, limits=BOX)


def test_limits_may_be_omitted(points):
    X, Y, Z = points
    fitted = meshsearch.MeshGrid(X, Y, Z, MPS)
    assert fitted.lims[0][0] <= X.min()
    assert fitted.lims[0][1] >= X.max()
    # each side is a whole number of cells
    for axis in range(3):
        span = fitted.lims[axis][1] - fitted.lims[axis][0]
        assert span / fitted.cellsize == pytest.approx(fitted.n_cells[axis])


# --------------------------------------------------------------------------
# exceptions


def test_error_derives_from_value_error():
    assert issubclass(meshsearch.Error, ValueError)


def test_index_error_derives_from_python_index_error():
    assert issubclass(meshsearch.IndexError, IndexError)


def test_index_error_also_derives_from_meshsearch_error():
    """The C++ hierarchy is mirrored: IndexError is an Error there too."""
    assert issubclass(meshsearch.IndexError, meshsearch.Error)


@pytest.mark.parametrize(
    "base", [lambda: meshsearch.IndexError, lambda: meshsearch.Error,
             lambda: IndexError, lambda: ValueError, lambda: LookupError]
)
def test_an_index_error_is_catchable_through_every_base(grids, base):
    """catch (const Error&) catches an IndexError in C++; so must except here."""
    with pytest.raises(base()):
        grids["matched"].nearest_object(10 ** 6)


def test_catching_meshsearch_error_catches_both_kinds(grids):
    grid = grids["matched"]
    caught = []
    for call in (lambda: grid.nearest_object(10 ** 6),        # IndexError
                 lambda: grid.nearest_object(5.0, 5.0, 5.0)):  # plain Error
        try:
            call()
        except meshsearch.Error as e:
            caught.append(type(e).__name__)
    assert caught == ["IndexError", "Error"]


def test_a_plain_error_is_not_an_index_error(grids):
    """The inheritance goes one way only."""
    grid = grids["matched"]
    for call in (lambda: grid.nearest_object(5.0, 5.0, 5.0),
                 lambda: grid.close_objects(0.5, 0.5, 0.5, -1.0)):
        with pytest.raises(meshsearch.Error) as caught:
            call()
        assert not isinstance(caught.value, meshsearch.IndexError)
        assert not isinstance(caught.value, IndexError)


def test_error_is_catchable_as_its_python_base(grids):
    with pytest.raises(ValueError):
        grids["matched"].nearest_object(5.0, 5.0, 5.0)


def test_index_error_is_catchable_as_its_python_base(grids):
    with pytest.raises(IndexError):
        grids["matched"].nearest_object(10**6)


@pytest.mark.parametrize(
    "call",
    [
        lambda g: g.nearest_object(10**6),
        lambda g: g.nearest_objects(1, 10**6),
        lambda g: g.close_objects(10**6, 1.0),
    ],
)
def test_bad_index_raises_index_error(grids, call):
    with pytest.raises(meshsearch.IndexError):
        call(grids["matched"])


def test_remove_object_raises_index_error(points):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, MPS, limits=BOX)
    with pytest.raises(meshsearch.IndexError):
        grid.remove_object(10**6)


def test_a_bad_count_is_not_an_index_error(grids):
    """N is a count, not an index, so it raises Error and not IndexError."""
    grid = grids["matched"]
    with pytest.raises(meshsearch.Error):
        grid.nearest_objects(len(grid) + 1, 0.5, 0.5, 0.5)
    with pytest.raises(meshsearch.Error):
        grid.nearest_objects(len(grid), 0)
    for call in (lambda: grid.nearest_objects(len(grid) + 1, 0.5, 0.5, 0.5),
                 lambda: grid.nearest_objects(len(grid), 0)):
        with pytest.raises(Exception) as caught:
            call()
        assert not isinstance(caught.value, meshsearch.IndexError)


@pytest.mark.parametrize(
    "call",
    [
        lambda g: g.close_objects(0.5, 0.5, 0.5, -1.0),
        lambda g: g.close_objects(0.5, 0.5, 0.5, 1.0, -1.0),
        lambda g: g.close_objects(0.5, 0.5, 0.5, 1.0, 2.0),
        lambda g: g.close_objects(0.5, 0.5, 0.5, float("nan")),
        lambda g: g.close_objects(0, float("nan")),
        lambda g: g.nearest_object(5.0, 0.5, 0.5),
        lambda g: g.get_cell(0.5, 5.0, 0.5),
        lambda g: g.get_cell_coords(0.5, 0.5, 5.0),
        lambda g: g.get_objects_in_cell(-1.0, 0.5, 0.5),
    ],
)
def test_documented_errors_raise(grids, call):
    with pytest.raises(meshsearch.Error):
        call(grids["matched"])


def test_add_object_outside_the_box_raises(points):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, MPS, limits=BOX)
    with pytest.raises(meshsearch.Error):
        grid.add_object(5.0, 0.5, 0.5)


def test_infinite_rmax_selects_everything(grids):
    grid = grids["matched"]
    assert len(grid.close_objects(0.5, 0.5, 0.5, float("inf"))) == len(grid)


# --------------------------------------------------------------------------
# the queries, against brute force


@pytest.mark.parametrize("regime", [name for name, _ in REGIMES])
def test_nearest_object_matches_brute_force(grids, points, regime):
    X, Y, Z = points
    grid = grids[regime]
    alive = np.ones(len(X), dtype=bool)
    rng = np.random.default_rng(1)

    for _ in range(40):
        x, y, z = rng.random(3)
        got = grid.nearest_object(x, y, z)
        assert distances(x, y, z, X, Y, Z)[got] == pytest.approx(
            brute_nearest(x, y, z, X, Y, Z, alive))


@pytest.mark.parametrize("regime", [name for name, _ in REGIMES])
def test_nearest_objects_matches_brute_force(grids, points, regime):
    X, Y, Z = points
    grid = grids[regime]
    alive = np.ones(len(X), dtype=bool)
    rng = np.random.default_rng(2)

    for n in (1, 7, 25):
        for _ in range(12):
            x, y, z = rng.random(3)
            got = grid.nearest_objects(n, x, y, z)
            assert len(got) == n
            assert len(set(got.tolist())) == n
            d = distances(x, y, z, X, Y, Z)[got]
            assert np.all(np.diff(d) >= -1e-12)          # nearest first
            assert d == pytest.approx(brute_ranked(x, y, z, X, Y, Z, alive)[:n])
            assert got.tolist() == brute_order(x, y, z, X, Y, Z, alive)[:n].tolist()


@pytest.mark.parametrize("regime", [name for name, _ in REGIMES])
def test_close_objects_matches_brute_force(grids, points, regime):
    X, Y, Z = points
    grid = grids[regime]
    alive = np.ones(len(X), dtype=bool)
    rng = np.random.default_rng(3)

    for rmax, rmin in [(0.05, 0.0), (0.2, 0.0), (0.2, 0.1), (0.45, 0.3)]:
        for _ in range(10):
            x, y, z = rng.random(3)
            got = set(grid.close_objects(x, y, z, rmax, rmin).tolist())
            assert got == brute_shell(x, y, z, X, Y, Z, alive, rmax, rmin)


@pytest.mark.parametrize("regime", [name for name, _ in REGIMES])
def test_index_overloads_exclude_the_query_object(grids, points, regime):
    X, Y, Z = points
    grid = grids[regime]
    alive = np.ones(len(X), dtype=bool)
    rng = np.random.default_rng(4)

    for index in rng.integers(0, len(X), 25):
        index = int(index)
        x, y, z = X[index], Y[index], Z[index]

        nearest = grid.nearest_object(index)
        assert nearest != index
        assert distances(x, y, z, X, Y, Z)[nearest] == pytest.approx(
            brute_nearest(x, y, z, X, Y, Z, alive, skip=index))

        near = grid.nearest_objects(5, index)
        assert index not in near.tolist()
        assert distances(x, y, z, X, Y, Z)[near] == pytest.approx(
            brute_ranked(x, y, z, X, Y, Z, alive, skip=index)[:5])

        shell = set(grid.close_objects(index, 0.2, 0.05).tolist())
        assert shell == brute_shell(x, y, z, X, Y, Z, alive, 0.2, 0.05, skip=index)


# --------------------------------------------------------------------------
# the guarantees that are not about a single query


def test_indices_are_stable_across_removals(points):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, MPS, limits=BOX)
    alive = np.ones(len(X), dtype=bool)

    removed = list(range(3, len(X), 7))
    for index in removed:
        assert grid.is_alive(index)
        grid.remove_object(index)
        alive[index] = False
    assert len(grid) == int(alive.sum())

    for index in removed:
        assert not grid.is_alive(index)

    rng = np.random.default_rng(5)
    for _ in range(25):
        x, y, z = rng.random(3)
        got = grid.nearest_object(x, y, z)
        assert alive[got]
        assert distances(x, y, z, X, Y, Z)[got] == pytest.approx(
            brute_nearest(x, y, z, X, Y, Z, alive))
        assert set(grid.close_objects(x, y, z, 0.2).tolist()) == brute_shell(
            x, y, z, X, Y, Z, alive, 0.2, 0.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda g, i: g.nearest_object(i),
        lambda g, i: g.nearest_objects(1, i),
        lambda g, i: g.close_objects(i, 0.1),
        lambda g, i: g.remove_object(i),
    ],
)
def test_a_removed_index_raises_from_every_member_taking_one(points, call):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, MPS, limits=BOX)
    grid.remove_object(11)
    with pytest.raises(meshsearch.IndexError):
        call(grid, 11)


def test_add_object_never_reuses_an_index(points):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, MPS, limits=BOX)

    for index in (0, 5, 9):
        grid.remove_object(index)

    highest = len(X) - 1
    rng = np.random.default_rng(6)
    for _ in range(20):
        x, y, z = rng.random(3)
        new = grid.add_object(x, y, z)
        assert new > highest
        highest = new
        assert grid.is_alive(new)
        for index in (0, 5, 9):
            assert not grid.is_alive(index)


def test_a_point_on_the_upper_face_lands_in_the_last_cell():
    grid = meshsearch.MeshGrid([0.5], [0.5], [0.5], 1.0,
                               limits=[[0.0, 10.0], [0.0, 4.0], [0.0, 2.0]])
    assert grid.n_cells == (10, 4, 2)
    assert grid.get_cell_coords(10.0, 4.0, 2.0) == (9, 3, 1)
    assert grid.get_cell_coords(0.0, 0.0, 0.0) == (0, 0, 0)

    added = grid.add_object(10.0, 4.0, 2.0)
    assert grid.get_objects_in_cell(10.0, 4.0, 2.0).tolist() == [added]
    assert grid.nearest_object(10.0, 4.0, 2.0) == added


def test_rmin_and_rmax_are_both_inclusive():
    # distances 1, 2 and 3 along x and 2, 3 along y from the first object, all
    # exact in binary so neither shell boundary is decided by rounding
    X = [5.0, 6.0, 7.0, 8.0, 5.0, 5.0]
    Y = [5.0, 5.0, 5.0, 5.0, 7.0, 8.0]
    Z = [5.0] * 6
    grid = meshsearch.MeshGrid(X, Y, Z, 1.0, limits=[[0.0, 16.0]] * 3)

    assert set(grid.close_objects(5.0, 5.0, 5.0, 2.0, 2.0).tolist()) == {2, 4}
    assert set(grid.close_objects(0, 2.0, 2.0).tolist()) == {2, 4}
    assert set(grid.close_objects(0, 3.0, 2.0).tolist()) == {2, 3, 4, 5}
    assert set(grid.close_objects(0, 3.0, 3.0).tolist()) == {3, 5}

    # a ball takes in everything out to rmax, the objects at exactly rmax among
    # them, and from a point it takes in the object at d = 0 too
    assert set(grid.close_objects(0, 2.0).tolist()) == {1, 2, 4}
    assert set(grid.close_objects(5.0, 5.0, 5.0, 2.0).tolist()) == {0, 1, 2, 4}


def test_coincident_points():
    X = [2.0, 2.0, 2.0, 4.0]
    grid = meshsearch.MeshGrid(X, [2.0] * 3 + [2.0], [2.0] * 4, 1.0,
                               limits=[[0.0, 16.0]] * 3)

    # self-exclusion is by identity, not by distance: a point coincident with
    # the query object is a different object, at distance 0, and is returned;
    # the query object itself is not
    assert set(grid.close_objects(2.0, 2.0, 2.0, 0.0).tolist()) == {0, 1, 2}
    assert set(grid.close_objects(0, 0.0).tolist()) == {1, 2}
    assert grid.nearest_object(0) in (1, 2)
    assert grid.nearest_objects(2, 0).tolist() == [1, 2]
    assert grid.nearest_objects(3, 0).tolist()[2] == 3


# "nearest first; ties by ascending index", on integer lattices, where every
# squared distance is a small exact integer and most of them tie


def test_nearest_objects_tie_order_by_hand():
    # object a*9 + b*3 + c at (a, b, c). From the centre, object 13: six objects
    # at d^2 = 1 (4, 10, 12, 14, 16, 22) and twelve at d^2 = 2, the lowest of
    # them 1 and 3; asking for 6 + 2 cuts that shell, so the last two places go
    # to 1 and 3. From (0.5, 0.5, 0.5) the eight corners of the unit cube all
    # tie. Cellsize 1.5 puts several objects in a cell, so the grid's internal
    # order is not the index order.
    a, b, c = np.meshgrid(np.arange(3.0), np.arange(3.0), np.arange(3.0), indexing="ij")
    grid = meshsearch.MeshGrid(a.ravel(), b.ravel(), c.ravel(), 1.5, limits=[[0.0, 3.0]] * 3)

    assert grid.nearest_objects(8, 13).tolist() == [4, 10, 12, 14, 16, 22, 1, 3]
    assert grid.nearest_objects(9, 1.0, 1.0, 1.0).tolist() == [13, 4, 10, 12, 14, 16, 22, 1, 3]
    assert grid.nearest_objects(3, 0.5, 0.5, 0.5).tolist() == [0, 1, 3]
    assert grid.nearest_objects(8, 0.5, 0.5, 0.5).tolist() == [0, 1, 3, 4, 9, 10, 12, 13]
    batched = grid.nearest_objects(3, np.array([0.5, 1.0]), np.array([0.5, 1.0]),
                                   np.array([0.5, 1.0]))
    assert batched.tolist() == [[0, 1, 3], [13, 4, 10]]


@pytest.mark.parametrize("cellsize", [0.7, 1.0, 2.5, 4.0])
def test_nearest_objects_tie_order_on_a_shuffled_lattice(cellsize):
    # a 10^3 lattice whose indices are a random permutation of the sites, so
    # index order has nothing to do with position or cell order; then some
    # removals, and additions coincident with existing sites, whose higher
    # indices must come after the objects they tie with. The counts end inside
    # shells, so the N-th place is a tie, and include otswap's 113.
    side = 10
    rng = np.random.default_rng(8)
    site = rng.permutation(side ** 3)
    X = (site // (side * side)).astype(float)
    Y = (site // side % side).astype(float)
    Z = (site % side).astype(float)
    grid = meshsearch.MeshGrid(X, Y, Z, cellsize, limits=[[0.0, float(side)]] * 3)
    alive = np.ones(len(X), dtype=bool)

    for index in rng.choice(len(X), 30, replace=False):
        grid.remove_object(int(index))
        alive[index] = False
    for source in rng.integers(0, side ** 3, 30):
        assert grid.add_object(X[source], Y[source], Z[source]) == len(X)
        X, Y, Z = np.append(X, X[source]), np.append(Y, Y[source]), np.append(Z, Z[source])
        alive = np.append(alive, True)

    live = np.flatnonzero(alive)
    queries = 0.5 * rng.integers(0, 2 * side + 1, (25, 3))
    for (x, y, z), index in zip(queries, rng.choice(live, 25)):
        index = int(index)
        for n in (1, 2, 5, 7, 13, 20, 27, 60, 113):
            assert grid.nearest_objects(n, x, y, z).tolist() == \
                brute_order(x, y, z, X, Y, Z, alive)[:n].tolist()
            assert grid.nearest_objects(n, index).tolist() == \
                brute_order(X[index], Y[index], Z[index], X, Y, Z, alive, skip=index)[:n].tolist()

    expected = [brute_order(x, y, z, X, Y, Z, alive)[:27] for x, y, z in queries]
    assert grid.nearest_objects(27, *queries.T.copy()).tolist() == np.array(expected).tolist()


@pytest.mark.parametrize("cellsize", [0.3, 0.12, 0.05])
def test_close_objects_across_the_radius_to_cellsize_ratio(cellsize):
    """The Rmax bound scans layers 0 to floor(Rmax/cellsize), so that ratio is
    what decides how many layers are scanned and what has to be swept."""
    X, Y, Z = cloud(1500, seed=99)
    alive = np.ones(1500, dtype=bool)
    grid = meshsearch.MeshGrid(X, Y, Z, cellsize, limits=BOX)
    rng = np.random.default_rng(11)

    for ratio in (0.05, 0.25, 0.5, 0.99, 1.0, 1.01, 1.5, 2.0, 2.5, 4.0, 8.0):
        rmax = ratio * cellsize
        for _ in range(4):
            x, y, z = rng.random(3)
            assert set(grid.close_objects(x, y, z, rmax).tolist()) == brute_shell(
                x, y, z, X, Y, Z, alive, rmax, 0.0)
            assert set(grid.close_objects(x, y, z, rmax, 0.5 * rmax).tolist()) == brute_shell(
                x, y, z, X, Y, Z, alive, rmax, 0.5 * rmax)


def test_an_object_in_the_last_layer_the_rmax_bound_scans():
    """Query point 0.75 into cell 10; object at the low corner of cell 13, an
    offset of 3, gap 2, layer 2, exactly 2.25 away. Rmax = 2.25 makes layer 2
    the last one scanned, so the object sits on the shell's outer edge."""
    grid = meshsearch.MeshGrid([10.75, 13.0], [10.5, 10.5], [10.5, 10.5], 1.0,
                               limits=[[0.0, 20.0]] * 3)
    assert grid.get_cell_coords(10.75, 10.5, 10.5)[0] == 10
    assert grid.get_cell_coords(13.0, 10.5, 10.5)[0] == 13
    assert grid.close_objects(0, 2.25).tolist() == [1]
    assert grid.close_objects(0, 2.25, 2.25).tolist() == [1]
    assert grid.close_objects(0, 2.2421875).size == 0


# --------------------------------------------------------------------------
# the batched queries: each one answers exactly as the loop of single queries
# would, results in query order, and fails where the loop would fail


@pytest.fixture(scope="module")
def batch_queries():
    rng = np.random.default_rng(4242)
    pts = rng.random((250, 3))
    return (np.ascontiguousarray(pts[:, 0]),
            np.ascontiguousarray(pts[:, 1]),
            np.ascontiguousarray(pts[:, 2]))


@pytest.mark.parametrize("cellsize", [0.25, 0.08])
def test_batched_nearest_object_matches_the_loop(points, batch_queries, cellsize):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, cellsize, limits=BOX)
    qx, qy, qz = batch_queries

    got = grid.nearest_object(qx, qy, qz)
    expected = np.array([grid.nearest_object(a, b, c) for a, b, c in zip(qx, qy, qz)],
                        dtype=np.uint32)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("cellsize", [0.25, 0.08])
@pytest.mark.parametrize("n", [1, 8])
def test_batched_nearest_objects_matches_the_loop(points, batch_queries, cellsize, n):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, cellsize, limits=BOX)
    qx, qy, qz = batch_queries

    got = grid.nearest_objects(n, qx, qy, qz)
    expected = np.array([grid.nearest_objects(n, a, b, c) for a, b, c in zip(qx, qy, qz)],
                        dtype=np.uint32)
    assert got.shape == (len(qx), n)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("cellsize", [0.25, 0.08])
@pytest.mark.parametrize("rmax,rmin", [(0.1, 0.0), (0.15, 0.08), (0.001, 0.0)])
def test_batched_close_objects_matches_the_loop(points, batch_queries, cellsize, rmax, rmin):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, cellsize, limits=BOX)
    qx, qy, qz = batch_queries

    indices, offsets = grid.close_objects(qx, qy, qz, rmax, rmin)
    for i, (a, b, c) in enumerate(zip(qx, qy, qz)):
        assert np.array_equal(indices[offsets[i]:offsets[i + 1]],
                              grid.close_objects(a, b, c, rmax, rmin))


def test_csr_invariants(grids, batch_queries):
    qx, qy, qz = batch_queries
    for rmax in (0.001, 0.05, 0.2):
        indices, offsets = grids["matched"].close_objects(qx, qy, qz, rmax)

        assert len(offsets) == len(qx) + 1
        assert offsets[0] == 0
        assert offsets[-1] == len(indices)
        assert np.all(np.diff(offsets.astype(np.int64)) >= 0)

        # a query with no results is an empty slice, never a missing entry
        counts = np.diff(offsets)
        for i in np.flatnonzero(counts == 0):
            assert indices[offsets[i]:offsets[i + 1]].size == 0


def test_csr_counts_come_from_np_diff(grids, batch_queries):
    qx, qy, qz = batch_queries
    indices, offsets = grids["matched"].close_objects(qx, qy, qz, 0.1)
    per_query = np.diff(offsets)
    assert per_query.sum() == len(indices)
    assert np.array_equal(
        per_query,
        [grids["matched"].close_objects(a, b, c, 0.1).size for a, b, c in zip(qx, qy, qz)])


def test_batched_results_are_uint32_and_offsets_int64(grids, batch_queries):
    qx, qy, qz = batch_queries
    grid = grids["matched"]
    indices, offsets = grid.close_objects(qx, qy, qz, 0.1)
    for array in (grid.nearest_object(qx, qy, qz),
                  grid.nearest_objects(4, qx, qy, qz),
                  indices):
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.uint32
    # offsets count results, which can outnumber what 32 bits address; int64
    # is numpy's standard integer, which slices and repeats without a cast
    assert isinstance(offsets, np.ndarray)
    assert offsets.dtype == np.int64


def test_an_empty_query_array(grids):
    grid = grids["matched"]
    empty = np.empty(0)

    assert grid.nearest_object(empty, empty, empty).shape == (0,)
    assert grid.nearest_objects(4, empty, empty, empty).shape == (0, 4)

    indices, offsets = grid.close_objects(empty, empty, empty, 0.1)
    assert indices.shape == (0,)
    assert offsets.shape == (1,)          # n_query + 1
    assert offsets[0] == 0


def test_a_single_point_array_is_still_the_batched_path(grids):
    grid = grids["matched"]
    one = np.array([0.5])
    got = grid.nearest_object(one, one, one)
    assert isinstance(got, np.ndarray) and got.shape == (1,)
    assert got[0] == grid.nearest_object(0.5, 0.5, 0.5)


def test_scalar_calls_are_unaffected_by_the_overload(grids):
    grid = grids["matched"]
    assert isinstance(grid.nearest_object(0.5, 0.5, 0.5), int)
    assert grid.nearest_objects(3, 0.5, 0.5, 0.5).shape == (3,)
    assert grid.close_objects(0.5, 0.5, 0.5, 0.1).ndim == 1


def test_every_index_query_has_a_batched_form(grids):
    """An array of indices selects the batch; a single index the scalar call."""
    grid = grids["matched"]
    indices = np.array([0, 1, 2], dtype=np.uint32)
    assert grid.nearest_object(indices).shape == (3,)
    assert grid.nearest_objects(3, indices).shape == (3, 3)
    found, offsets = grid.close_objects(indices, 0.1)
    assert found.dtype == np.uint32 and offsets.dtype == np.int64 and offsets.shape == (4,)
    assert grid.count_objects(indices, 0.1).shape == (3,)
    assert isinstance(grid.nearest_object(1), int)
    assert grid.nearest_objects(3, 1).shape == (3,)


@pytest.mark.parametrize(
    "call",
    [
        lambda g, bad, y, z: g.nearest_object(bad, y, z),
        lambda g, bad, y, z: g.nearest_objects(3, bad, y, z),
        lambda g, bad, y, z: g.close_objects(bad, y, z, 0.1),
    ],
)
def test_a_batched_call_raises_naming_the_query(grids, batch_queries, call):
    qx, qy, qz = batch_queries
    bad = qx.copy()
    bad[17] = 5.0                                  # outside the box

    with pytest.raises(meshsearch.Error) as caught:
        call(grids["matched"], bad, qy, qz)
    assert "query 17" in str(caught.value)


def test_a_batched_call_raises_on_the_first_bad_query(grids, batch_queries):
    qx, qy, qz = batch_queries
    bad = qx.copy()
    bad[9] = 5.0
    bad[40] = 5.0

    with pytest.raises(meshsearch.Error) as caught:
        grids["matched"].nearest_object(bad, qy, qz)
    assert "query 9" in str(caught.value)


def test_a_bad_radius_still_raises_in_a_batched_call(grids, batch_queries):
    qx, qy, qz = batch_queries
    for rmax, rmin in [(-1.0, 0.0), (1.0, 2.0), (float("nan"), 0.0)]:
        with pytest.raises(meshsearch.Error):
            grids["matched"].close_objects(qx, qy, qz, rmax, rmin)


def test_mismatched_query_array_lengths_raise(grids, batch_queries):
    qx, qy, qz = batch_queries
    grid = grids["matched"]
    with pytest.raises(meshsearch.Error):
        grid.nearest_object(qx[:-1], qy, qz)
    with pytest.raises(meshsearch.Error):
        grid.nearest_objects(3, qx, qy[:-1], qz)
    with pytest.raises(meshsearch.Error):
        grid.close_objects(qx, qy, qz[:-1], 0.1)


def test_batched_queries_see_removals(points, batch_queries):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, 0.08, limits=BOX)
    qx, qy, qz = batch_queries

    before = grid.nearest_object(qx, qy, qz)
    for index in np.unique(before)[:20]:
        grid.remove_object(int(index))
    after = grid.nearest_object(qx, qy, qz)

    assert np.array_equal(
        after,
        np.array([grid.nearest_object(a, b, c) for a, b, c in zip(qx, qy, qz)],
                 dtype=np.uint32))
    assert all(grid.is_alive(int(i)) for i in after)


# --------------------------------------------------------------------------
# count_objects: the number close_objects would return, checked against brute
# force in every regime, scalar and batched, point and index forms


@pytest.mark.parametrize("name", [name for name, _ in REGIMES])
@pytest.mark.parametrize("rmax,rmin", [(0.1, 0.0), (0.15, 0.08), (0.001, 0.0)])
def test_count_objects_matches_brute_force(points, grids, batch_queries, name, rmax, rmin):
    X, Y, Z = points
    grid = grids[name]
    alive = np.ones(len(X), dtype=bool)
    qx, qy, qz = batch_queries

    expected = [len(brute_shell(a, b, c, X, Y, Z, alive, rmax, rmin))
                for a, b, c in zip(qx, qy, qz)]
    assert [grid.count_objects(a, b, c, rmax, rmin) for a, b, c in zip(qx, qy, qz)] == expected
    assert grid.count_objects(qx, qy, qz, rmax, rmin).tolist() == expected

    objects = np.arange(0, len(X), 7)
    expected = [len(brute_shell(X[i], Y[i], Z[i], X, Y, Z, alive, rmax, rmin, skip=i))
                for i in objects]
    assert [grid.count_objects(int(i), rmax, rmin) for i in objects] == expected
    assert grid.count_objects(objects, rmax, rmin).tolist() == expected


def test_count_objects_result_types(grids, batch_queries):
    qx, qy, qz = batch_queries
    grid = grids["matched"]
    assert isinstance(grid.count_objects(0.5, 0.5, 0.5, 0.1), int)
    assert isinstance(grid.count_objects(3, 0.1), int)
    for counts in (grid.count_objects(qx, qy, qz, 0.1),
                   grid.count_objects(np.arange(10, dtype=np.uint32), 0.1)):
        assert isinstance(counts, np.ndarray)
        assert counts.dtype == np.int64


def test_count_objects_takes_any_integer_index_array(grids):
    grid = grids["matched"]
    expected = [grid.count_objects(i, 0.1) for i in (3, 1, 4)]
    for dtype in (np.uint32, np.int64, np.int32, np.uint64, np.uint16):
        assert grid.count_objects(np.array([3, 1, 4], dtype=dtype), 0.1).tolist() == expected
    # strided input is read where it lies, not required to be contiguous
    assert grid.count_objects(np.array([3, 0, 1, 0, 4])[::2], 0.1).tolist() == expected


def test_count_objects_never_wraps_an_index(grids):
    """A value that is not a valid index raises: none is wrapped to 32 bits,
    which would silently name another object."""
    grid = grids["matched"]
    for bad in (np.array([1, -1]), np.array([1, 2 ** 32 + 1], dtype=np.uint64)):
        with pytest.raises(meshsearch.IndexError) as caught:
            grid.count_objects(bad, 0.1)
        assert "query 1" in str(caught.value)
    with pytest.raises(TypeError):
        grid.count_objects(np.array([1.0, 2.0]), 0.1)


def test_count_objects_index_batch_raises_on_a_removed_object(points):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, MPS, limits=BOX)
    grid.remove_object(5)
    with pytest.raises(meshsearch.IndexError) as caught:
        grid.count_objects(np.array([1, 2, 5, 6], dtype=np.uint32), 0.1)
    assert "query 2" in str(caught.value)


def test_count_objects_bad_radii_raise(grids, batch_queries):
    qx, qy, qz = batch_queries
    grid = grids["matched"]
    for rmax, rmin in [(-1.0, 0.0), (1.0, 2.0), (float("nan"), 0.0)]:
        for call in (lambda: grid.count_objects(0.5, 0.5, 0.5, rmax, rmin),
                     lambda: grid.count_objects(3, rmax, rmin),
                     lambda: grid.count_objects(qx, qy, qz, rmax, rmin),
                     lambda: grid.count_objects(np.arange(3), rmax, rmin)):
            with pytest.raises(meshsearch.Error):
                call()


# --------------------------------------------------------------------------
# batches over object indices: each answers as the loop of single index
# queries would, in order, on a grid with removals so that the indices are not
# simply the positions in the grid; and they share count_objects' input rules


@pytest.fixture(scope="module")
def thinned(points):
    X, Y, Z = points
    out = {}
    for name, size in REGIMES:
        grid = meshsearch.MeshGrid(X, Y, Z, size, limits=BOX)
        for index in range(0, len(X), 9):
            grid.remove_object(index)
        out[name] = grid
    return out


def live_objects(grid, n):
    return np.array([i for i in range(n) if grid.is_alive(i)][::5], dtype=np.uint32)


@pytest.mark.parametrize("name", [name for name, _ in REGIMES])
def test_index_batches_match_the_loop(points, thinned, name):
    grid = thinned[name]
    objects = live_objects(grid, len(points[0]))

    assert grid.nearest_object(objects).tolist() == [grid.nearest_object(int(i)) for i in objects]

    got = grid.nearest_objects(6, objects)
    assert got.shape == (len(objects), 6)
    assert np.array_equal(got, np.array([grid.nearest_objects(6, int(i)) for i in objects]))

    for rmax, rmin in [(0.1, 0.0), (0.15, 0.08)]:
        found, offsets = grid.close_objects(objects, rmax, rmin)
        assert offsets[0] == 0 and offsets[-1] == len(found) and len(offsets) == len(objects) + 1
        for q, i in enumerate(objects):
            assert np.array_equal(found[offsets[q]:offsets[q + 1]],
                                  grid.close_objects(int(i), rmax, rmin))
        assert np.array_equal(np.diff(offsets), grid.count_objects(objects, rmax, rmin))


@pytest.mark.parametrize("dtype", [np.uint32, np.int64, np.int32, np.uint64, np.int16])
def test_index_batches_take_any_integer_dtype(thinned, dtype):
    grid = thinned["matched"]
    objects = np.array([3, 1, 4], dtype=dtype)
    assert grid.nearest_object(objects).tolist() == [grid.nearest_object(i) for i in (3, 1, 4)]
    assert np.array_equal(grid.nearest_objects(2, objects),
                          np.array([grid.nearest_objects(2, i) for i in (3, 1, 4)]))
    found, offsets = grid.close_objects(objects, 0.1)
    assert np.array_equal(found, np.concatenate([grid.close_objects(i, 0.1) for i in (3, 1, 4)]))


@pytest.mark.parametrize("call", [
    lambda g, idx: g.nearest_object(idx),
    lambda g, idx: g.nearest_objects(3, idx),
    lambda g, idx: g.close_objects(idx, 0.1),
    lambda g, idx: g.count_objects(idx, 0.1),
])
def test_index_batches_never_wrap_and_name_the_query(thinned, call):
    grid = thinned["matched"]
    for bad in (np.array([1, 2, -1]), np.array([1, 2, 2 ** 32 + 1], dtype=np.uint64),
                np.array([1, 2, 9], dtype=np.uint32)):          # 9 is removed
        with pytest.raises(meshsearch.IndexError) as caught:
            call(grid, bad)
        assert "query 2" in str(caught.value)
    with pytest.raises(TypeError):
        call(grid, np.array([1.0, 2.0]))


def test_nearest_objects_index_batch_out(thinned):
    grid = thinned["matched"]
    objects = np.array([1, 2, 3, 4], dtype=np.int64)
    out = np.zeros((4, 5), dtype=np.uint32)
    assert grid.nearest_objects(5, objects, out=out) is out
    assert np.array_equal(out, grid.nearest_objects(5, objects))
    with pytest.raises(meshsearch.Error):
        grid.nearest_objects(5, objects, out=np.zeros((4, 4), dtype=np.uint32))
    with pytest.raises(TypeError):
        grid.nearest_objects(5, objects, out=np.zeros((4, 5), dtype=np.int64))


def test_index_batches_refuse_what_the_grid_refuses_even_when_empty(thinned):
    grid = thinned["matched"]
    none = np.empty(0, dtype=np.uint32)
    assert grid.nearest_object(none).shape == (0,)
    assert grid.nearest_objects(3, none).shape == (0, 3)
    found, offsets = grid.close_objects(none, 0.1)
    assert found.shape == (0,) and offsets.tolist() == [0]
    with pytest.raises(meshsearch.Error):
        grid.nearest_objects(len(grid), none)            # n must be below the count
    with pytest.raises(meshsearch.Error):
        grid.close_objects(none, 0.1, 0.2)


def test_a_large_index_batch_returns_results_in_query_order(points, grids):
    grid = grids["dense"]
    rng = np.random.default_rng(6)
    objects = rng.integers(0, len(points[0]), 110_000)

    def chunks(call):
        return [call(objects[i:i + 1000]) for i in range(0, len(objects), 1000)]

    assert np.array_equal(grid.nearest_object(objects), np.concatenate(chunks(grid.nearest_object)))
    assert np.array_equal(grid.nearest_objects(3, objects),
                          np.concatenate(chunks(lambda o: grid.nearest_objects(3, o))))
    found, offsets = grid.close_objects(objects, 0.05)
    parts = chunks(lambda o: grid.close_objects(o, 0.05))
    assert np.array_equal(found, np.concatenate([p[0] for p in parts]))
    assert np.array_equal(np.diff(offsets), np.concatenate([np.diff(p[1]) for p in parts]))


# --------------------------------------------------------------------------
# out=: the caller's array is filled and returned, and must already be exactly
# the result, since a converted copy would leave the caller's array unwritten


def test_out_is_filled_and_returned(grids, batch_queries):
    qx, qy, qz = batch_queries
    grid = grids["matched"]

    out = np.zeros((len(qx), 4), dtype=np.uint32)
    assert grid.nearest_objects(4, qx, qy, qz, out=out) is out
    assert np.array_equal(out, grid.nearest_objects(4, qx, qy, qz))

    counts = np.zeros(len(qx), dtype=np.int64)
    assert grid.count_objects(qx, qy, qz, 0.1, out=counts) is counts
    assert np.array_equal(counts, grid.count_objects(qx, qy, qz, 0.1))

    objects = np.arange(20)
    counts = np.zeros(20, dtype=np.int64)
    assert grid.count_objects(objects, 0.1, out=counts) is counts
    assert np.array_equal(counts, grid.count_objects(objects, 0.1))


def test_nearest_object_out_is_filled_returned_and_checked(grids, batch_queries):
    qx, qy, qz = batch_queries
    grid = grids["matched"]
    objects = np.arange(20)

    out = np.zeros(len(qx), dtype=np.uint32)
    assert grid.nearest_object(qx, qy, qz, out=out) is out
    assert np.array_equal(out, grid.nearest_object(qx, qy, qz))

    out = np.zeros(20, dtype=np.uint32)
    assert grid.nearest_object(objects, out=out) is out
    assert np.array_equal(out, grid.nearest_object(objects))

    untouched = np.full(len(qx), 7, dtype=np.uint32)
    for bad in (np.zeros(len(qx), dtype=np.int64), [0] * len(qx)):
        with pytest.raises(TypeError):
            grid.nearest_object(qx, qy, qz, out=bad)
    with pytest.raises(meshsearch.Error):
        grid.nearest_object(qx, qy, qz, out=np.zeros(len(qx) + 1, dtype=np.uint32))
    with pytest.raises(meshsearch.Error):
        grid.nearest_object(qx, qy, qz, out=np.zeros((len(qx), 2), dtype=np.uint32)[:, 0])
    bad_x = qx.copy()
    bad_x[3] = 1e9
    with pytest.raises(meshsearch.Error):
        grid.nearest_object(bad_x, qy, qz, out=untouched)
    assert (untouched == 7).all()


def test_out_may_be_reused_across_calls(grids, batch_queries):
    qx, qy, qz = batch_queries
    grid = grids["matched"]
    counts = np.empty(len(qx), dtype=np.int64)
    for rmax in (0.05, 0.1, 0.2):
        grid.count_objects(qx, qy, qz, rmax, out=counts)
        assert np.array_equal(counts, grid.count_objects(qx, qy, qz, rmax))


def test_out_may_be_a_view_into_a_larger_array(grids, batch_queries):
    """A C-contiguous view is accepted and only its own elements written."""
    qx, qy, qz = batch_queries
    grid = grids["matched"]
    table = np.zeros((3, len(qx)), dtype=np.int64)
    grid.count_objects(qx, qy, qz, 0.1, out=table[1])
    assert np.array_equal(table[1], grid.count_objects(qx, qy, qz, 0.1))
    assert not table[0].any() and not table[2].any()


@pytest.mark.parametrize("make", [
    lambda n: np.zeros(n, dtype=np.uint64),        # wrong dtype, even unsigned
    lambda n: np.zeros(n, dtype=np.uint32),        # wrong dtype
    lambda n: [0] * n,                             # not an array
])
def test_out_of_the_wrong_type_raises_type_error(grids, batch_queries, make):
    qx, qy, qz = batch_queries
    with pytest.raises(TypeError):
        grids["matched"].count_objects(qx, qy, qz, 0.1, out=make(len(qx)))


def test_a_read_only_out_raises_type_error(grids, batch_queries):
    qx, qy, qz = batch_queries
    out = np.zeros(len(qx), dtype=np.int64)
    out.flags.writeable = False
    with pytest.raises(TypeError):
        grids["matched"].count_objects(qx, qy, qz, 0.1, out=out)


@pytest.mark.parametrize("make", [
    lambda n: np.zeros(n - 1, dtype=np.int64),          # too short
    lambda n: np.zeros((n, 1), dtype=np.int64),         # wrong rank
    lambda n: np.zeros(2 * n, dtype=np.int64)[::2],     # not contiguous
])
def test_out_of_the_wrong_shape_or_layout_raises(grids, batch_queries, make):
    qx, qy, qz = batch_queries
    with pytest.raises(meshsearch.Error):
        grids["matched"].count_objects(qx, qy, qz, 0.1, out=make(len(qx)))


def test_nearest_objects_out_must_match_both_dimensions(grids, batch_queries):
    qx, qy, qz = batch_queries
    grid = grids["matched"]
    for shape in [(len(qx), 3), (len(qx) - 1, 4), (4, len(qx))]:
        with pytest.raises(meshsearch.Error):
            grid.nearest_objects(4, qx, qy, qz, out=np.zeros(shape, dtype=np.uint32))
    with pytest.raises(meshsearch.Error):
        grid.nearest_objects(4, qx, qy, qz,
                             out=np.zeros((4, len(qx)), dtype=np.uint32).T)
    with pytest.raises(TypeError):
        grid.nearest_objects(4, qx, qy, qz, out=np.zeros((len(qx), 4), dtype=np.uint64))


def test_out_is_untouched_when_the_call_raises(grids, batch_queries):
    qx, qy, qz = batch_queries
    bad = qx.copy()
    bad[17] = 5.0
    counts = np.full(len(qx), 7, dtype=np.int64)
    with pytest.raises(meshsearch.Error):
        grids["matched"].count_objects(bad, qy, qz, 0.1, out=counts)
    assert np.all(counts == 7)


def test_out_is_keyword_only(grids, batch_queries):
    qx, qy, qz = batch_queries
    with pytest.raises(TypeError):
        grids["matched"].count_objects(qx, qy, qz, 0.1, 0.0,
                                       np.zeros(len(qx), dtype=np.int64))


# --------------------------------------------------------------------------
# a large batch is answered in cell order internally; its results must still
# come back in query order, identical to the same queries asked in small
# batches, which run in input order


@pytest.fixture(scope="module")
def large_batch():
    rng = np.random.default_rng(99)
    pts = rng.random((120_000, 3))
    return tuple(np.ascontiguousarray(pts[:, a]) for a in range(3))


def chunked(call, queries, size=1000):
    qx, qy, qz = queries
    return [call(qx[i:i + size], qy[i:i + size], qz[i:i + size])
            for i in range(0, len(qx), size)]


@pytest.mark.parametrize("name", ["dense", "sparse"])
def test_a_large_batch_returns_results_in_query_order(grids, large_batch, name):
    grid = grids[name]
    qx, qy, qz = large_batch

    assert np.array_equal(grid.nearest_object(qx, qy, qz),
                          np.concatenate(chunked(grid.nearest_object, large_batch)))
    assert np.array_equal(grid.nearest_objects(3, qx, qy, qz),
                          np.concatenate(chunked(lambda a, b, c: grid.nearest_objects(3, a, b, c),
                                                 large_batch)))
    assert np.array_equal(grid.count_objects(qx, qy, qz, 0.05),
                          np.concatenate(chunked(lambda a, b, c: grid.count_objects(a, b, c, 0.05),
                                                 large_batch)))

    indices, offsets = grid.close_objects(qx, qy, qz, 0.05)
    parts = chunked(lambda a, b, c: grid.close_objects(a, b, c, 0.05), large_batch)
    assert np.array_equal(indices, np.concatenate([p[0] for p in parts]))
    assert np.array_equal(np.diff(offsets), np.concatenate([np.diff(p[1]) for p in parts]))


def test_a_large_index_batch_returns_counts_in_query_order(points, grids):
    grid = grids["dense"]
    rng = np.random.default_rng(5)
    objects = rng.integers(0, len(points[0]), 110_000)
    counts = grid.count_objects(objects, 0.05)
    expected = np.concatenate([grid.count_objects(objects[i:i + 1000], 0.05)
                               for i in range(0, len(objects), 1000)])
    assert np.array_equal(counts, expected)


# --------------------------------------------------------------------------
# the GIL is released during a batched call, and const queries may run
# concurrently on one grid, so threads querying one grid each get the answer
# a lone call gets


def test_batched_calls_from_several_threads(grids, large_batch):
    import threading

    grid = grids["matched"]
    qx, qy, qz = (a[:30_000] for a in large_batch)
    expected = (grid.nearest_objects(3, qx, qy, qz), grid.count_objects(qx, qy, qz, 0.05))
    failures = []

    def work():
        for _ in range(3):
            if not (np.array_equal(grid.nearest_objects(3, qx, qy, qz), expected[0])
                    and np.array_equal(grid.count_objects(qx, qy, qz, 0.05), expected[1])):
                failures.append(True)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not failures


# --------------------------------------------------------------------------
# free-threaded builds: importing the module must not bring the GIL back, and
# the guarantee that const members may run concurrently on one grid is then
# exercised with no GIL to serialise the single-point calls


@pytest.mark.skipif(not hasattr(sys, "_is_gil_enabled"), reason="not a CPython with the GIL switch")
def test_import_leaves_the_gil_disabled_on_free_threaded_builds():
    if sysconfig.get_config_var("Py_GIL_DISABLED") != 1:
        pytest.skip("regular build")
    code = "import sys, meshsearch; sys.exit(1 if sys._is_gil_enabled() else 0)"
    result = subprocess.run([sys.executable, "-c", code], env={k: v for k, v in os.environ.items() if k != "PYTHON_GIL"},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def run_threads(work, n_threads=6):
    barrier = threading.Barrier(n_threads)
    failures = []

    def target():
        barrier.wait()
        try:
            work()
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=target) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not failures, failures[0]


def every_query(grid, qx, qy, qz, objects):
    """Every query kind, single-point and batched, by point and by index."""
    nq = 200
    out = []
    for i in range(0, nq, 7):
        x, y, z = float(qx[i]), float(qy[i]), float(qz[i])
        out += [grid.nearest_object(x, y, z), grid.nearest_objects(4, x, y, z),
                grid.close_objects(x, y, z, 0.08), grid.close_objects(x, y, z, 0.08, 0.03),
                grid.count_objects(x, y, z, 0.08), grid.contains(x, y, z)]
        j = int(objects[i])
        out += [grid.nearest_object(j), grid.nearest_objects(4, j),
                grid.close_objects(j, 0.08), grid.count_objects(j, 0.08), grid.is_alive(j)]
    q = (qx[:nq], qy[:nq], qz[:nq])
    out += [grid.nearest_object(*q), grid.nearest_objects(4, *q),
            grid.close_objects(*q, 0.08), grid.close_objects(*q, 0.08, 0.03),
            grid.count_objects(*q, 0.08), grid.contains(*q),
            grid.nearest_object(objects[:nq]), grid.nearest_objects(4, objects[:nq]),
            grid.close_objects(objects[:nq], 0.08), grid.count_objects(objects[:nq], 0.08),
            grid.is_alive(objects[:nq])]
    return out


def same(a, b):
    if isinstance(a, tuple):
        return all(same(u, v) for u, v in zip(a, b))
    return np.array_equal(a, b)


@pytest.mark.parametrize("periodic", [False, (True, False, True), True], ids=["open", "mixed", "periodic"])
@pytest.mark.parametrize("regime", ["dense", "matched", "sparse"])
def test_threads_querying_one_grid_get_the_serial_results(points, periodic, regime):
    size = dict(REGIMES)[regime]
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, size, limits=BOX, periodic=periodic)
    rng = np.random.default_rng(7)
    qx, qy, qz = rng.random((3, 200))
    objects = rng.integers(0, len(X), 200).astype(np.uint32)
    expected = every_query(grid, qx, qy, qz, objects)

    def work():
        for _ in range(3):
            got = every_query(grid, qx, qy, qz, objects)
            assert len(got) == len(expected)
            assert all(same(g, e) for g, e in zip(got, expected))

    run_threads(work)


def test_a_wide_shell_at_large_rmin():
    # the configuration random data does not reach: the query point on the low
    # corner of its cell and the object just inside the far corner of the cell
    # six cells away on each axis, at 7*sqrt(3) = 12.124 cellsize
    grid = meshsearch.MeshGrid([10.0, 17.0 - 1e-9], [10.0, 17.0 - 1e-9],
                               [10.0, 17.0 - 1e-9], 1.0, limits=[[0.0, 40.0]] * 3)
    assert grid.close_objects(10.0, 10.0, 10.0, 12.2, 12.0).tolist() == [1]
    assert grid.close_objects(0, 12.2, 12.0).tolist() == [1]


# --------------------------------------------------------------------------
# constructor validation, zero-extent axes, and grids too large to count;
# the same guarantees as the C++ groups testConstructorValidation,
# testZeroExtent, testCountLimits and testLongAxis


NAN, INF = float("nan"), float("inf")
SMALL = ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
SMALL_BOX = [[0.0, 4.0], [0.0, 4.0], [0.0, 4.0]]


def raises_plain_error(build):
    with pytest.raises(meshsearch.Error) as caught:
        build()
    assert not isinstance(caught.value, meshsearch.IndexError)


@pytest.mark.parametrize("cellsize", [INF, -INF, NAN, 0.0, -1.0])
def test_a_cellsize_not_finite_and_positive_raises(cellsize):
    raises_plain_error(lambda: meshsearch.MeshGrid(*SMALL, cellsize, limits=SMALL_BOX))
    raises_plain_error(lambda: meshsearch.MeshGrid(*SMALL, cellsize))


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
@pytest.mark.parametrize("end", [0, 1])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_limits_not_finite_raise(axis, end, bad):
    limits = [list(side) for side in SMALL_BOX]
    limits[axis][end] = bad
    raises_plain_error(lambda: meshsearch.MeshGrid(*SMALL, 1.0, limits=limits))
    raises_plain_error(lambda: meshsearch.MeshGrid([], [], [], 1.0, limits=limits))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_inverted_limits_raise(axis):
    limits = [list(side) for side in SMALL_BOX]
    limits[axis] = [4.0, 0.0]
    raises_plain_error(lambda: meshsearch.MeshGrid(*SMALL, 1.0, limits=limits))
    raises_plain_error(lambda: meshsearch.MeshGrid([], [], [], 1.0, limits=limits))


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_coordinate_not_finite_raises(axis, bad):
    coords = [list(c) for c in SMALL]
    coords[axis][1] = bad
    raises_plain_error(lambda: meshsearch.MeshGrid(*coords, 1.0, limits=SMALL_BOX))
    raises_plain_error(lambda: meshsearch.MeshGrid(*coords, 1.0))


# Points all equal along one axis (flat) or two (needle), built with no limits
# and with lo == hi limits: one cell on each such axis, a box one cellsize wide
# centred on it, and the same answers as brute force.
@pytest.mark.parametrize("flat_axes", [1, 2])
def test_a_zero_extent_axis_gets_one_cell(flat_axes):
    rng = np.random.default_rng(31337)
    level, cellsize, n = 0.37, 0.1, 800
    X = rng.random(n)
    Y = np.full(n, level) if flat_axes == 2 else rng.random(n)
    Z = np.full(n, level)
    alive = np.ones(n, dtype=bool)
    limits = [[0.0, 1.0], [level, level] if flat_axes == 2 else [0.0, 1.0], [level, level]]

    for grid in (meshsearch.MeshGrid(X, Y, Z, cellsize),
                 meshsearch.MeshGrid(X, Y, Z, cellsize, limits=limits)):
        for axis in range(3 - flat_axes, 3):
            assert grid.n_cells[axis] == 1
            assert grid.lims[axis][0] == pytest.approx(level - cellsize / 2)
            assert grid.lims[axis][1] == pytest.approx(level + cellsize / 2)

        for _ in range(40):
            x = rng.uniform(grid.lims[0][0], grid.lims[0][1])
            y = level if flat_axes == 2 else rng.uniform(grid.lims[1][0], grid.lims[1][1])
            got = grid.nearest_object(x, y, level)
            assert distances(x, y, level, X[got], Y[got], Z[got]) == pytest.approx(
                brute_nearest(x, y, level, X, Y, Z, alive))
            got = set(grid.close_objects(x, y, level, 0.08).tolist())
            assert got == brute_shell(x, y, level, X, Y, Z, alive, 0.08, 0.0)


# Each would need terabytes or more; raising at all, quickly and without
# MemoryError, shows that the counts are checked before anything is allocated.
@pytest.mark.parametrize("limits,cellsize", [
    (SMALL_BOX, 1e-300),
    ([[0.0, 2147483648.0], [0.0, 1.0], [0.0, 1.0]], 1.0),
    ([[0.0, 4194304.0]] * 3, 1.0),
])
def test_a_grid_too_large_to_count_raises(limits, cellsize):
    raises_plain_error(lambda: meshsearch.MeshGrid([], [], [], cellsize, limits=limits))


# 46341 cells along x: past the 32-bit threshold where 0.1.0's offset
# arithmetic overflowed, in a grid of a few MB. Two points at the two ends make
# the nearest-object search walk the whole length.
def test_a_long_axis_past_the_32_bit_threshold():
    length = 46341.0
    limits = [[0.0, length], [0.0, 1.0], [0.0, 1.0]]
    rng = np.random.default_rng(46341)
    X, Y, Z = rng.uniform(0, length, 3000), rng.random(3000), rng.random(3000)
    alive = np.ones(3000, dtype=bool)
    grid = meshsearch.MeshGrid(X, Y, Z, 1.0, limits=limits)
    assert grid.n_cells[0] == 46341
    for _ in range(20):
        x, y, z = rng.uniform(0, length), rng.random(), rng.random()
        got = grid.nearest_object(x, y, z)
        assert distances(x, y, z, X[got], Y[got], Z[got]) == pytest.approx(
            brute_nearest(x, y, z, X, Y, Z, alive))
        assert set(grid.close_objects(x, y, z, 40.0).tolist()) == brute_shell(
            x, y, z, X, Y, Z, alive, 40.0, 0.0)

    ends = meshsearch.MeshGrid([0.25, length - 0.25], [0.5, 0.5], [0.5, 0.5], 1.0, limits=limits)
    assert ends.nearest_object(0) == 1
    assert ends.nearest_object(1) == 0


# --------------------------------------------------------------------------
# rounding at cell faces, and the extremes the constructor accepts
#
# A point within a few ulp of a cell face can be assigned to the cell beyond
# it. The header still promises exact shell membership and the true nearest
# object; each case below is one where a rule with no allowance for this lost
# a result. They mirror the C++ suite's cases exactly.

FACE_BOX = [[0.0, 0.8], [0.0, 0.0], [0.0, 0.0]]
FACE_Q = float(np.nextafter(0.4, 0.0))          # in cell 3, just below 0.4


def test_an_object_at_exactly_rmax_beyond_a_rounded_face():
    """0.7 lands in cell 7 although the double lies just below 7 * 0.1; from
    the query, 0.7 - q is the double 0.3 exactly, so the object is at Rmax."""
    assert 0.7 - FACE_Q == 0.3
    grid = meshsearch.MeshGrid([0.7], [0.0], [0.0], 0.1, limits=FACE_BOX)
    assert grid.get_cell_coords(0.7, 0.0, 0.0)[0] == 7
    assert grid.close_objects(FACE_Q, 0.0, 0.0, 0.3).tolist() == [0]
    assert grid.close_objects(FACE_Q, 0.0, 0.0, 0.3, 0.3).tolist() == [0]

    indices, offsets = grid.close_objects(np.array([FACE_Q]), np.zeros(1), np.zeros(1), 0.3)
    assert indices.tolist() == [0] and offsets.tolist() == [0, 1]

    pair = meshsearch.MeshGrid([FACE_Q, 0.7], [0.0, 0.0], [0.0, 0.0], 0.1, limits=FACE_BOX)
    assert pair.close_objects(0, 0.3).tolist() == [1]


def test_the_nearest_object_beyond_a_rounded_face():
    """A, just below 0.1, is in shell 2 from the query; B, at 0.7, is in
    shell 3 and nearer. A stop with no margin returns A."""
    a = float.fromhex("0x1.9999999999996p-4")
    assert (0.7 - FACE_Q) ** 2 < (FACE_Q - a) ** 2
    grid = meshsearch.MeshGrid([a, 0.7], [0.0, 0.0], [0.0, 0.0], 0.1, limits=FACE_BOX)
    assert grid.nearest_object(FACE_Q, 0.0, 0.0) == 1
    assert grid.nearest_object(np.array([FACE_Q]), np.zeros(1), np.zeros(1)).tolist() == [1]
    assert grid.nearest_objects(1, np.array([FACE_Q]), np.zeros(1), np.zeros(1)).tolist() == [[1]]

    three = meshsearch.MeshGrid([a, 0.7, FACE_Q], [0.0] * 3, [0.0] * 3, 0.1, limits=FACE_BOX)
    assert three.nearest_object(2) == 1


def test_an_object_at_exactly_rmin_in_an_overhanging_top_cell():
    """A cellsize of about one ulp of coordinates near -9.8e7: the top cell
    overhangs its nominal face by most of a cell, and a range starting at
    floor(Rmin/cellsize) - 4 never reaches the object."""
    h = float.fromhex
    c = h("0x1.457ef4d38d4ebp-26")
    limits = [[h("-0x1.74a7f8cac18dp+26"), h("-0x1.74a7f8cac18bep+26")],
              [h("-0x1.74a7f8cac1876p+26"), h("-0x1.74a7f8cac1868p+26")],
              [h("-0x1.74a7f8cac18cp+26"), h("-0x1.74a7f8cac18aep+26")]]
    obj = [h("-0x1.74a7f8cac18cdp+26"), h("-0x1.74a7f8cac1874p+26"), h("-0x1.74a7f8cac18bep+26")]
    q = [h("-0x1.74a7f8cac18bdp+26"), h("-0x1.74a7f8cac1867p+26"), h("-0x1.74a7f8cac18adp+26")]
    rmin = h("0x1.ab884ef702284p-22")

    grid = meshsearch.MeshGrid([obj[0]], [obj[1]], [obj[2]], c, limits=limits)
    assert grid.close_objects(*q, INF, rmin).tolist() == [0]
    indices, _ = grid.close_objects(*(np.array([v]) for v in q), INF, rmin)
    assert indices.tolist() == [0]


def test_the_grown_box_contains_the_limits():
    """(cells * cellsize - range) / 2 came out negative here, so the grown box
    lay inside the limits and a point on a limit was refused."""
    c, hi = 5.078493365404232, 3279294.892895552
    grid = meshsearch.MeshGrid([0.0, hi], [0.5, 0.5], [0.5, 0.5], c,
                               limits=[[0.0, hi], [0.0, 1.0], [0.0, 1.0]])
    assert grid.lims[0][0] <= 0.0 and grid.lims[0][1] >= hi
    assert grid.nearest_object(0.0, 0.5, 0.5) == 0
    assert grid.nearest_object(hi, 0.5, 0.5) == 1


@pytest.mark.parametrize("build", [
    lambda: meshsearch.MeshGrid([0.0, 9e159], [0.0, 0.0], [0.0, 0.0], 1e159),
    lambda: meshsearch.MeshGrid([], [], [], 1e153, limits=[[-1e154, 1e154], [0.0, 0.0], [0.0, 0.0]]),
])
def test_a_box_whose_squared_diagonal_overflows_raises(build):
    raises_plain_error(build)


def test_a_box_whose_squared_diagonal_is_representable_is_accepted():
    grid = meshsearch.MeshGrid([0.0, 6e153], [0.0, 6e153], [0.0, 6e153], 1e153,
                               limits=[[0.0, 6e153]] * 3)
    assert grid.nearest_object(0) == 1
    assert grid.close_objects(0, INF).tolist() == [1]


@pytest.mark.parametrize("cellsize", [1e-310, 5e-324])
def test_a_cellsize_whose_reciprocal_overflows_raises(cellsize):
    raises_plain_error(lambda: meshsearch.MeshGrid([0.0], [0.0], [0.0], cellsize))


def test_a_batched_nearest_objects_with_too_large_n_raises_before_allocating():
    """n_query * n would be far beyond any memory; the first query refuses n
    before the result array is sized, so this is the single-point error, not
    a MemoryError."""
    grid = meshsearch.MeshGrid(*SMALL, 1.0, limits=SMALL_BOX)
    q = np.full(1_000_000, 1.0)
    with pytest.raises(meshsearch.Error) as caught:
        grid.nearest_objects(2**32 - 1, q, q, q)
    assert "query 0" in str(caught.value)


# --------------------------------------------------------------------------
# periodic axes
#
# The geometry is checked exhaustively by the C++ suite; these check what the
# binding adds: the keyword-only signature, the two forms of `periodic`, the
# read-only property, the effective cellsize, and one brute-force comparison in
# the minimum-image metric.


def test_limits_and_periodic_are_keyword_only():
    X, Y, Z = SMALL
    with pytest.raises(TypeError):
        meshsearch.MeshGrid(X, Y, Z, 1.0, SMALL_BOX)
    with pytest.raises(TypeError):
        meshsearch.MeshGrid(X, Y, Z, 1.0, SMALL_BOX, True)


def test_periodic_defaults_to_false_and_is_a_tuple_of_bools():
    grid = meshsearch.MeshGrid(*SMALL, 1.0)
    assert grid.periodic == (False, False, False)
    assert all(type(p) is bool for p in grid.periodic)


@pytest.mark.parametrize("flag, expected", [
    (True, (True, True, True)),
    (False, (False, False, False)),
    ((True, False, True), (True, False, True)),
    ([False, True, False], (False, True, False)),
    (np.array([True, True, False]), (True, True, False)),
])
def test_periodic_takes_a_bool_or_a_triple(flag, expected):
    grid = meshsearch.MeshGrid(*SMALL, 1.0, limits=SMALL_BOX, periodic=flag)
    assert grid.periodic == expected


def test_periodic_is_read_only():
    grid = meshsearch.MeshGrid(*SMALL, 1.0, limits=SMALL_BOX, periodic=True)
    with pytest.raises(AttributeError):
        grid.periodic = (False, False, False)


@pytest.mark.parametrize("flag", [(True, False), (True,) * 4, "xyz", 1, None])
def test_periodic_of_the_wrong_form_is_a_type_error(flag):
    with pytest.raises(TypeError):
        meshsearch.MeshGrid(*SMALL, 1.0, limits=SMALL_BOX, periodic=flag)


def test_a_periodic_axis_without_limits_raises():
    raises_plain_error(lambda: meshsearch.MeshGrid(*SMALL, 1.0, periodic=True))
    raises_plain_error(lambda: meshsearch.MeshGrid(*SMALL, 1.0, periodic=(False, True, False)))


def test_periodic_sides_must_be_multiples_of_the_shortest():
    raises_plain_error(lambda: meshsearch.MeshGrid(
        [0.5], [0.5], [0.5], 1.0, limits=[[0.0, 4.0], [0.0, 6.5], [0.0, 4.0]], periodic=True))


def test_the_periodic_box_is_exactly_the_limits_and_the_cellsize_divides_it():
    box = [[0.0, 1.0], [0.0, 1.0], [0.0, 2.0]]
    grid = meshsearch.MeshGrid([0.5], [0.5], [0.5], 0.3, limits=box, periodic=True)
    assert np.array_equal(grid.lims, np.array(box))
    assert grid.cellsize <= 0.3
    assert grid.n_cells == (4, 4, 8)
    assert grid.cellsize == 0.25


def test_periodic_queries_match_brute_force_in_the_minimum_image_metric():
    rng = np.random.default_rng(5)
    X, Y, Z = rng.random(400), rng.random(400), rng.random(400)
    grid = meshsearch.MeshGrid(X, Y, Z, 0.15, limits=[[0.0, 1.0]] * 3, periodic=(True, False, True))

    def image(d, periodic):
        return np.where(periodic, d - np.where(d >= 0.5, 1.0, 0.0) + np.where(d < -0.5, 1.0, 0.0), d)

    for _ in range(40):
        q = rng.random(3)
        d = np.sqrt(image(X - q[0], True) ** 2 + (Y - q[1]) ** 2 + image(Z - q[2], True) ** 2)
        got = grid.nearest_objects(7, *q)
        assert np.allclose(np.sort(d)[:7], d[got])
        inside = grid.close_objects(*q, 0.3)
        assert sorted(inside.tolist()) == sorted(np.nonzero(d <= 0.3)[0].tolist())
        assert grid.close_objects(*q, np.inf).size == 400


# --------------------------------------------------------------------------
# the 1.0 interface: version, repr, Batch, array-likes, negative arguments,
# batched is_alive and contains, copies
#
# The header says every member taking an index raises on one that is out of
# range or removed, and that errors are meshsearch.Error. A negative index is
# out of range, so it is an IndexError and never wraps to a large unsigned
# one; a negative count is invalid input, so it is an Error. is_alive is the
# one member taking an index that does not raise.


def test_version_is_the_one_cmake_generates():
    import re
    from pathlib import Path
    assert isinstance(meshsearch.__version__, str)
    cmake = Path(__file__).resolve().parent.parent / "CMakeLists.txt"
    if not cmake.exists():
        pytest.skip("run from outside the source tree")
    declared = re.search(r"project\(meshsearch VERSION ([0-9.]+)", cmake.read_text()).group(1)
    assert meshsearch.__version__ == declared


def test_repr_names_the_grid(points):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, 0.25, limits=BOX, periodic=(True, False, False))
    grid.remove_object(0)
    assert repr(grid) == (
        f"MeshGrid(objects={len(X) - 1}, n_cells={grid.n_cells}, "
        f"cellsize={grid.cellsize!r}, periodic=(True, False, False))")
    assert str(grid) == repr(grid)


def test_close_objects_returns_the_named_tuple(grids, batch_queries):
    grid = grids["matched"]
    qx, qy, qz = batch_queries
    for found in (grid.close_objects(qx, qy, qz, 0.1),
                  grid.close_objects(np.arange(5, dtype=np.uint32), 0.1)):
        assert isinstance(found, meshsearch.Batch)
        assert isinstance(found, tuple) and len(found) == 2
        assert found._fields == ("indices", "offsets")
        indices, offsets = found
        assert found.indices is indices and found.offsets is offsets
        assert indices.dtype == np.uint32 and offsets.dtype == np.int64
    assert meshsearch.Batch.__module__ == "meshsearch"
    import pickle
    again = pickle.loads(pickle.dumps(found))
    assert np.array_equal(again.indices, found.indices) and np.array_equal(again.offsets, found.offsets)


@pytest.mark.parametrize("call", [
    lambda g, i: g.nearest_object(i),
    lambda g, i: g.nearest_objects(3, i),
    lambda g, i: g.close_objects(i, 0.1),
    lambda g, i: g.count_objects(i, 0.1),
    lambda g, i: g.remove_object(i),
])
@pytest.mark.parametrize("bad", [-1, -(2 ** 31), 2 ** 32, 2 ** 70, -(2 ** 70), np.int64(-3)])
def test_a_single_index_that_is_not_valid_raises_index_error(points, call, bad):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, 0.1, limits=BOX)
    with pytest.raises(meshsearch.IndexError):
        call(grid, bad)
    with pytest.raises(IndexError):
        call(grid, bad)
    assert len(grid) == len(X)


def test_a_single_index_of_the_wrong_type_is_a_type_error(grids):
    for bad in (1.0, "1", None):
        with pytest.raises(TypeError):
            grids["matched"].nearest_object(bad)
        with pytest.raises(TypeError):
            grids["matched"].count_objects(bad, 0.1)


def test_a_negative_index_does_not_wrap_to_the_last_object(points):
    """-1 is not the last object, as it would be for a list, and not 2**32 - 1."""
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, 0.1, limits=BOX)
    assert grid.is_alive(len(X) - 1)
    with pytest.raises(meshsearch.IndexError, match="-1"):
        grid.nearest_object(-1)


@pytest.mark.parametrize("bad", [-1, -(2 ** 40)])
def test_a_negative_n_raises_error_in_every_form(grids, batch_queries, bad):
    grid = grids["matched"]
    qx, qy, qz = batch_queries
    calls = [
        lambda: grid.nearest_objects(bad, 0.5, 0.5, 0.5),
        lambda: grid.nearest_objects(bad, 3),
        lambda: grid.nearest_objects(bad, qx, qy, qz),
        lambda: grid.nearest_objects(bad, np.arange(4, dtype=np.uint32)),
        lambda: grid.nearest_objects(bad, qx[:0], qy[:0], qz[:0]),
        lambda: grid.nearest_objects(bad, np.arange(0, dtype=np.uint32)),
    ]
    for call in calls:
        with pytest.raises(meshsearch.Error) as caught:
            call()
        assert not isinstance(caught.value, meshsearch.IndexError)
        assert "negative" in str(caught.value)


def test_an_n_beyond_32_bits_raises_error(grids):
    with pytest.raises(meshsearch.Error):
        grids["matched"].nearest_objects(2 ** 40, 0.5, 0.5, 0.5)
    with pytest.raises(TypeError):
        grids["matched"].nearest_objects(2.0, 0.5, 0.5, 0.5)


# Batch inputs accept any array-like. A float64 or integer array is read where
# it is, a list or a float32 array is converted first, and either way the
# answer is that of the float64 array of the same values.

ARRAY_LIKES = [
    ("list", lambda a: a.tolist()),
    ("tuple", lambda a: tuple(a.tolist())),
    ("float32-exact", lambda a: a.astype(np.float32).astype(np.float64).astype(np.float32)),
    ("strided", lambda a: np.repeat(a, 2)[::2]),
    ("negative-stride", lambda a: a[::-1].copy()[::-1]),
    ("read-only", lambda a: a),
]


@pytest.mark.parametrize("name,convert", ARRAY_LIKES, ids=[n for n, _ in ARRAY_LIKES])
def test_batch_coordinates_accept_any_array_like(points, name, convert):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, 0.1, limits=BOX)
    rng = np.random.default_rng(3)
    base = rng.random((3, 60))
    if name == "float32-exact":
        base = base.astype(np.float32).astype(np.float64)
    qx, qy, qz = (np.ascontiguousarray(row) for row in base)
    if name == "read-only":
        for row in (qx, qy, qz):
            row.setflags(write=False)
    cx, cy, cz = (convert(row) for row in (qx, qy, qz))

    assert np.array_equal(grid.nearest_object(cx, cy, cz), grid.nearest_object(qx, qy, qz))
    assert np.array_equal(grid.nearest_objects(4, cx, cy, cz), grid.nearest_objects(4, qx, qy, qz))
    got, want = grid.close_objects(cx, cy, cz, 0.1), grid.close_objects(qx, qy, qz, 0.1)
    assert np.array_equal(got.indices, want.indices) and np.array_equal(got.offsets, want.offsets)
    assert np.array_equal(grid.count_objects(cx, cy, cz, 0.1), grid.count_objects(qx, qy, qz, 0.1))
    assert np.array_equal(grid.contains(cx, cy, cz), grid.contains(qx, qy, qz))


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.int32, np.int64, np.uint64])
def test_batch_coordinates_accept_integer_arrays(dtype):
    grid = meshsearch.MeshGrid([0.0, 1.0, 2.0, 3.0], [0.0] * 4, [0.0] * 4, 1.0, limits=[[0, 4], [-1, 1], [-1, 1]])
    q = np.array([0, 1, 3], dtype=dtype)
    zero = np.zeros(3, dtype=dtype)
    assert grid.nearest_object(q, zero, zero).tolist() == [0, 1, 3]
    assert grid.count_objects(q, zero, zero, 1).tolist() == [2, 3, 2]
    assert grid.contains(q, zero, zero).tolist() == [True, True, True]


def test_batch_indices_accept_lists_and_tuples(thinned):
    grid = thinned["matched"]
    wanted = [3, 1, 4, 10]
    for given in (wanted, tuple(wanted), np.array(wanted), range(3, 5)):
        listed = list(given)
        assert grid.nearest_object(given).tolist() == [grid.nearest_object(i) for i in listed]
        assert np.array_equal(grid.nearest_objects(2, given), [grid.nearest_objects(2, i) for i in listed])
        assert grid.count_objects(given, 0.1).tolist() == [grid.count_objects(i, 0.1) for i in listed]
        assert grid.is_alive(given).tolist() == [grid.is_alive(i) for i in listed]


def test_empty_batches_of_any_kind(grids):
    grid = grids["matched"]
    assert grid.nearest_object([], [], []).shape == (0,)
    assert grid.nearest_object([]).shape == (0,)
    assert grid.nearest_objects(3, []).shape == (0, 3)
    assert grid.count_objects([], 0.1).shape == (0,)
    assert grid.is_alive([]).shape == (0,)
    assert grid.contains([], [], []).shape == (0,)
    found = grid.close_objects([], 0.1)
    assert found.indices.shape == (0,) and found.offsets.tolist() == [0]


def test_batch_inputs_of_the_wrong_kind_or_shape_raise(grids):
    grid = grids["matched"]
    with pytest.raises(TypeError):
        grid.nearest_object(["a", "b"], ["a", "b"], ["a", "b"])
    with pytest.raises(TypeError):
        grid.nearest_object(np.array([1 + 2j]), np.array([0.5]), np.array([0.5]))
    with pytest.raises(TypeError):
        grid.nearest_object(np.array([True, False]))
    with pytest.raises(meshsearch.Error, match="one-dimensional"):
        grid.nearest_object(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(meshsearch.Error, match="one-dimensional"):
        grid.nearest_object(np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(meshsearch.Error, match="differ in length"):
        grid.nearest_object([0.5, 0.5], [0.5], [0.5])


def test_batch_indices_from_a_list_are_range_checked(thinned):
    grid = thinned["matched"]
    with pytest.raises(meshsearch.IndexError, match="query 1"):
        grid.nearest_object([1, -1])
    with pytest.raises(meshsearch.IndexError, match="query 1"):
        grid.nearest_object([1, 2 ** 40])
    with pytest.raises(meshsearch.IndexError, match="query 1"):
        grid.close_objects([1, 9], 0.1)


def test_a_scalar_is_still_not_a_batch(grids):
    grid = grids["matched"]
    assert isinstance(grid.nearest_object(0.5, 0.5, 0.5), int)
    assert isinstance(grid.count_objects(0.5, 0.5, 0.5, 0.1), int)
    assert isinstance(grid.nearest_object(np.float64(0.5), np.float64(0.5), np.float64(0.5)), int)
    assert isinstance(grid.nearest_object(np.int64(3)), int)
    assert grid.nearest_objects(3, 0.5, 0.5, 0.5).shape == (3,)
    assert grid.close_objects(0.5, 0.5, 0.5, 0.1).ndim == 1
    assert grid.close_objects(3, 0.1).ndim == 1
    assert isinstance(grid.contains(0.5, 0.5, 0.5), bool)
    assert isinstance(grid.is_alive(3), bool)
    assert grid.nearest_object([0.5], [0.5], [0.5]).shape == (1,)
    assert grid.nearest_object([3]).shape == (1,)


def test_is_alive_batch_matches_the_scalar_and_never_raises(thinned):
    grid = thinned["matched"]
    n = 2000
    indices = np.concatenate([np.arange(0, n + 5), [-1, -(2 ** 40), 2 ** 32, 2 ** 33, 4294967295]])
    got = grid.is_alive(indices)
    assert got.dtype == np.bool_ and got.shape == indices.shape
    assert got.tolist() == [grid.is_alive(int(i)) for i in indices]
    assert got[:n].sum() == len(grid) and not got[n:].any()
    assert grid.is_alive(np.array([5, 6], dtype=np.uint8)).tolist() == [True, True]
    assert grid.is_alive(np.array([0, 9, 18], dtype=np.int8)).tolist() == [False] * 3
    assert grid.is_alive(-1) is False and grid.is_alive(2 ** 70) is False
    with pytest.raises(TypeError):
        grid.is_alive(np.array([1.0]))
    with pytest.raises(TypeError):
        grid.is_alive(1.0)


def test_is_alive_follows_removal_and_addition(points):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X[:10], Y[:10], Z[:10], 0.2, limits=BOX)
    grid.remove_object(2)
    new = grid.add_object(0.5, 0.5, 0.5)
    assert grid.is_alive(np.arange(12)).tolist() == [i != 2 and i != 11 or i == new for i in range(12)]


def test_contains_matches_whether_a_query_is_accepted(points):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, 0.2, limits=BOX)
    rng = np.random.default_rng(11)
    q = rng.uniform(-0.5, 1.5, size=(500, 3))
    lo, hi = grid.lims[:, 0], grid.lims[:, 1]
    edge = np.array([[lo[0], lo[1], lo[2]], [hi[0], hi[1], hi[2]], [np.nan, 0.5, 0.5],
                     [0.5, np.inf, 0.5], [0.5, 0.5, -np.inf], [hi[0] + 1e-9, 0.5, 0.5]])
    q = np.vstack([q, edge])

    def accepted(p):
        try:
            grid.nearest_object(*p)
            return True
        except meshsearch.Error:
            return False

    got = grid.contains(q[:, 0].copy(), q[:, 1].copy(), q[:, 2].copy())
    assert got.dtype == np.bool_
    assert got.tolist() == [accepted(p) for p in q] == [grid.contains(*p) for p in q]
    assert 0 < got.sum() < len(q)


def test_contains_and_is_alive_out(points):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, 0.2, limits=BOX)
    rng = np.random.default_rng(12)
    q = rng.uniform(-0.5, 1.5, size=(40, 3))
    out = np.empty(40, dtype=np.bool_)
    got = grid.contains(q[:, 0].copy(), q[:, 1].copy(), q[:, 2].copy(), out=out)
    assert got is out
    assert out.tolist() == [grid.contains(*p) for p in q]

    out[:] = True
    indices = np.arange(-20, 20)
    assert grid.is_alive(indices, out=out) is out
    assert out.tolist() == [grid.is_alive(int(i)) for i in indices]

    for call in (lambda o: grid.contains(q[:, 0].copy(), q[:, 1].copy(), q[:, 2].copy(), out=o),
                 lambda o: grid.is_alive(indices, out=o)):
        with pytest.raises(TypeError):
            call(np.empty(40, dtype=np.uint8))
        with pytest.raises(TypeError):
            call([True] * 40)
        with pytest.raises(meshsearch.Error):
            call(np.empty(41, dtype=np.bool_))
        with pytest.raises(meshsearch.Error):
            call(np.empty(80, dtype=np.bool_)[::2])
        read_only = np.empty(40, dtype=np.bool_)
        read_only.setflags(write=False)
        with pytest.raises(TypeError):
            call(read_only)


def test_contains_arrays_of_unequal_length_raise(grids):
    with pytest.raises(meshsearch.Error, match="differ in length"):
        grids["matched"].contains([0.5, 0.5], [0.5], [0.5])


# Copies. The copy is independent of the original and answers every query as it
# does, the order within a cell included: that order is unspecified by the
# header, but a copy is the same grid, so it must not change. The comparison
# below covers every query kind and is written to take any two grids.


def same_answers(a, b, n_objects):
    rng = np.random.default_rng(77)
    q = rng.uniform(-0.1, 1.1, size=(120, 3))
    qx, qy, qz = (np.ascontiguousarray(q[:, k]) for k in range(3))
    inside = [tuple(p) for p in q if a.contains(*p)]
    assert inside

    assert len(a) == len(b) and a.cellsize == b.cellsize and a.n_cells == b.n_cells
    assert a.periodic == b.periodic and np.array_equal(a.lims, b.lims)
    assert repr(a) == repr(b)

    objects = np.arange(n_objects, dtype=np.int64)
    assert np.array_equal(a.is_alive(objects), b.is_alive(objects))
    assert np.array_equal(a.contains(qx, qy, qz), b.contains(qx, qy, qz))
    alive = [int(i) for i in objects[a.is_alive(objects)]]
    live = np.array(alive, dtype=np.uint32)

    for point in inside:
        assert a.nearest_object(*point) == b.nearest_object(*point)
        assert a.get_cell(*point) == b.get_cell(*point)
        assert a.get_cell_coords(*point) == b.get_cell_coords(*point)
        assert a.get_objects_in_cell(*point).tolist() == b.get_objects_in_cell(*point).tolist()
        for n in (1, 5, 40):
            assert a.nearest_objects(n, *point).tolist() == b.nearest_objects(n, *point).tolist()
        for rmax, rmin in ((0.05, 0.0), (0.2, 0.1), (np.inf, 0.0) if any(a.periodic) else (0.5, 0.0)):
            assert a.close_objects(*point, rmax, rmin).tolist() == b.close_objects(*point, rmax, rmin).tolist()
            assert a.count_objects(*point, rmax, rmin) == b.count_objects(*point, rmax, rmin)

    for index in alive[::7]:
        assert a.nearest_object(index) == b.nearest_object(index)
        assert a.nearest_objects(6, index).tolist() == b.nearest_objects(6, index).tolist()
        assert a.close_objects(index, 0.2, 0.05).tolist() == b.close_objects(index, 0.2, 0.05).tolist()
        assert a.count_objects(index, 0.2, 0.05) == b.count_objects(index, 0.2, 0.05)

    ix, iy, iz = (np.ascontiguousarray([p[k] for p in inside]) for k in range(3))
    assert np.array_equal(a.nearest_object(ix, iy, iz), b.nearest_object(ix, iy, iz))
    assert np.array_equal(a.nearest_objects(9, ix, iy, iz), b.nearest_objects(9, ix, iy, iz))
    assert np.array_equal(a.nearest_object(live), b.nearest_object(live))
    assert np.array_equal(a.nearest_objects(9, live), b.nearest_objects(9, live))
    for got, want in ((a.close_objects(ix, iy, iz, 0.2, 0.05), b.close_objects(ix, iy, iz, 0.2, 0.05)),
                      (a.close_objects(live, 0.2, 0.05), b.close_objects(live, 0.2, 0.05))):
        assert np.array_equal(got.indices, want.indices) and np.array_equal(got.offsets, want.offsets)
    assert np.array_equal(a.count_objects(ix, iy, iz, 0.2), b.count_objects(ix, iy, iz, 0.2))
    assert np.array_equal(a.count_objects(live, 0.2), b.count_objects(live, 0.2))


def worked_grid(**options):
    """A grid with removed and added objects, some added into cells that were
    full, so that its cells hold objects in more than one stretch."""
    X, Y, Z = cloud(1500, seed=5)
    grid = meshsearch.MeshGrid(X, Y, Z, 0.12, limits=BOX, **options)
    for index in range(2, 1500, 5):
        grid.remove_object(index)
    rng = np.random.default_rng(6)
    for point in rng.random((600, 3)):
        grid.add_object(*point)
    for index in range(1, 2100, 11):
        if grid.is_alive(index):
            grid.remove_object(index)
    return grid


@pytest.mark.parametrize("options", [{}, {"periodic": (True, False, True)}], ids=["plain", "periodic"])
@pytest.mark.parametrize("make", [copy.copy, copy.deepcopy, lambda g: copy.deepcopy({"g": g})["g"]],
                         ids=["copy", "deepcopy", "deepcopy-nested"])
def test_a_copy_answers_every_query_as_the_original_does(options, make):
    grid = worked_grid(**options)
    duplicate = make(grid)
    assert duplicate is not grid and type(duplicate) is meshsearch.MeshGrid
    same_answers(grid, duplicate, 2100)


def test_a_copy_is_independent_of_the_original():
    grid = worked_grid()
    reference = worked_grid()
    duplicate = copy.copy(grid)

    new = duplicate.add_object(0.5, 0.5, 0.5)
    assert len(duplicate) == len(grid) + 1 and not grid.is_alive(new)
    duplicate.remove_object(0)
    duplicate.remove_object(3)
    assert len(duplicate) == len(grid) - 1
    same_answers(grid, reference, 2100)

    again = copy.deepcopy(grid)
    grid.add_object(0.1, 0.2, 0.3)
    grid.remove_object(1500 + 1)
    assert len(again) == len(reference)
    same_answers(again, reference, 2100)


def test_a_copy_of_a_grid_keeps_issuing_the_next_index_the_original_would():
    grid = worked_grid()
    duplicate = copy.deepcopy(grid)
    assert duplicate.add_object(0.4, 0.4, 0.4) == grid.add_object(0.4, 0.4, 0.4)
    assert duplicate.add_object(0.6, 0.4, 0.4) == grid.add_object(0.6, 0.4, 0.4)
    same_answers(grid, duplicate, 2200)


def test_the_periodic_example_runs():
    import runpy
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "examples" / "example_periodic.py"
    runpy.run_path(str(path), run_name="__main__")


def test_the_batch_example_runs():
    import runpy
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "examples" / "example_batch.py"
    runpy.run_path(str(path), run_name="__main__")


def test_the_tour_example_runs():
    import runpy
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "examples" / "example.py"
    runpy.run_path(str(path), run_name="__main__")


# An out= array that is also the indices argument: the Python batches read
# their indices into their own copy first, so the answer is the one a separate
# array would receive. close_objects has no out=, so it cannot alias.


def test_out_may_be_the_indices_array(grids):
    grid = grids["matched"]
    objects = np.arange(40, dtype=np.uint32)

    same = objects.copy()
    assert grid.nearest_object(same, out=same) is same
    assert np.array_equal(same, grid.nearest_object(objects))

    wide = np.zeros(40 * 3, dtype=np.uint32)
    wide[:40] = objects
    out = wide.reshape(40, 3)
    assert grid.nearest_objects(3, wide[:40], out=out) is out
    assert np.array_equal(out, grid.nearest_objects(3, objects))

    counts = objects.astype(np.int64)
    assert grid.count_objects(counts, 0.1, out=counts) is counts
    assert np.array_equal(counts, grid.count_objects(objects, 0.1))


# --------------------------------------------------------------------------
# workers=: the batch forms may run on several threads; the results do not
# depend on how many, in content or order. A batch is split only when it is
# large enough for every thread to get some work, so these batches hold
# thousands of queries; the C++ suite checks the paths in detail.

WORKERS = (2, 3, 7, -1)


@pytest.fixture(scope="module")
def worker_grids(points):
    X, Y, Z = points
    return {"plain": meshsearch.MeshGrid(X, Y, Z, MPS, limits=BOX),
            "periodic": meshsearch.MeshGrid(X, Y, Z, 0.125, limits=BOX, periodic=True)}


def batch_results(grid, qx, qy, qz, objects, **kw):
    """Every batch form, returning and with out=, as a flat list of arrays."""
    n = len(qx)
    found = [grid.nearest_object(qx, qy, qz, **kw), grid.nearest_object(objects, **kw),
             grid.nearest_objects(5, qx, qy, qz, **kw), grid.nearest_objects(5, objects, **kw),
             grid.count_objects(qx, qy, qz, 0.08, **kw), grid.count_objects(objects, 0.08, 0.02, **kw)]
    found += list(grid.close_objects(qx, qy, qz, 0.08, **kw))
    found += list(grid.close_objects(objects, 0.08, 0.02, **kw))
    out = [np.zeros(n, np.uint32), np.zeros(len(objects), np.uint32), np.zeros((n, 5), np.uint32),
           np.zeros(n, np.int64), np.zeros(len(objects), np.int64)]
    grid.nearest_object(qx, qy, qz, out=out[0], **kw)
    grid.nearest_object(objects, out=out[1], **kw)
    grid.nearest_objects(5, qx, qy, qz, out=out[2], **kw)
    grid.count_objects(qx, qy, qz, 0.08, out=out[3], **kw)
    grid.count_objects(objects, 0.08, 0.02, out=out[4], **kw)
    return found + out


@pytest.mark.parametrize("kind", ["plain", "periodic"])
def test_workers_change_no_result(worker_grids, kind):
    grid = worker_grids[kind]
    qx, qy, qz = cloud(5000, seed=33)
    objects = np.random.default_rng(34).integers(0, len(grid), 5000)
    expected = batch_results(grid, qx, qy, qz, objects)
    assert all(np.array_equal(a, b) for a, b in zip(expected, batch_results(grid, qx, qy, qz, objects, workers=1)))
    for workers in WORKERS:
        got = batch_results(grid, qx, qy, qz, objects, workers=workers)
        assert len(got) == len(expected)
        for a, b in zip(expected, got):
            assert a.dtype == b.dtype and np.array_equal(a, b)


def test_workers_change_no_result_in_a_large_batch(worker_grids):
    # From 100000 queries a batch answers in cell order, and an index batch is
    # grouped by cell; neither depends on the number of workers either.
    grid = worker_grids["plain"]
    objects = np.random.default_rng(35).integers(0, len(grid), 110_000)
    qx, qy, qz = cloud(110_000, seed=36)
    expected = [grid.count_objects(objects, 0.06), *grid.close_objects(objects, 0.06),
                grid.nearest_objects(4, qx, qy, qz), *grid.close_objects(qx, qy, qz, 0.05)]
    for workers in (3, -1):
        got = [grid.count_objects(objects, 0.06, workers=workers), *grid.close_objects(objects, 0.06, workers=workers),
               grid.nearest_objects(4, qx, qy, qz, workers=workers),
               *grid.close_objects(qx, qy, qz, 0.05, workers=workers)]
        assert all(np.array_equal(a, b) for a, b in zip(expected, got))


def test_workers_is_keyword_only(worker_grids):
    grid = worker_grids["plain"]
    qx, qy, qz = cloud(600, seed=37)
    objects = np.arange(600)
    with pytest.raises(TypeError):
        grid.close_objects(qx, qy, qz, 0.1, 0.0, 2)
    with pytest.raises(TypeError):
        grid.count_objects(objects, 0.1, 0.0, None, 2)
    with pytest.raises(TypeError):
        grid.nearest_object(qx, qy, qz, None, 2)
    with pytest.raises(TypeError):
        grid.nearest_objects(3, objects, None, 2)


def test_single_queries_take_no_workers(worker_grids):
    grid = worker_grids["plain"]
    with pytest.raises(TypeError):
        grid.nearest_object(0.5, 0.5, 0.5, workers=2)
    with pytest.raises(TypeError):
        grid.nearest_objects(3, 7, workers=2)
    with pytest.raises(TypeError):
        grid.close_objects(7, 0.1, workers=2)
    with pytest.raises(TypeError):
        grid.count_objects(0.5, 0.5, 0.5, 0.1, workers=2)
    with pytest.raises(TypeError):
        grid.is_alive(np.arange(10), workers=2)
    with pytest.raises(TypeError):
        grid.contains(*cloud(10), workers=2)


@pytest.mark.parametrize("bad", [0, -2, -3, -(2 ** 31), -(10 ** 30)])
def test_workers_below_one_other_than_minus_one_raise(worker_grids, bad):
    grid = worker_grids["plain"]
    qx, qy, qz = cloud(600, seed=38)
    objects = np.arange(600)
    calls = [lambda: grid.nearest_object(qx, qy, qz, workers=bad),
             lambda: grid.nearest_object(objects, workers=bad),
             lambda: grid.nearest_objects(2, qx, qy, qz, workers=bad),
             lambda: grid.nearest_objects(2, objects, workers=bad),
             lambda: grid.close_objects(qx, qy, qz, 0.1, workers=bad),
             lambda: grid.close_objects(objects, 0.1, workers=bad),
             lambda: grid.count_objects(qx, qy, qz, 0.1, workers=bad),
             lambda: grid.count_objects(objects, 0.1, workers=bad)]
    for call in calls:
        with pytest.raises(meshsearch.Error) as raised:
            call()
        assert not isinstance(raised.value, meshsearch.IndexError)


@pytest.mark.parametrize("bad", [2.0, "2", None, [2]])
def test_workers_must_be_an_integer(worker_grids, bad):
    grid = worker_grids["plain"]
    qx, qy, qz = cloud(600, seed=39)
    with pytest.raises(TypeError):
        grid.close_objects(qx, qy, qz, 0.1, workers=bad)
    with pytest.raises(TypeError):
        grid.count_objects(np.arange(600), 0.1, workers=bad)


def test_more_workers_than_cores_are_allowed(worker_grids):
    grid = worker_grids["plain"]
    qx, qy, qz = cloud(3000, seed=40)
    expected = grid.count_objects(qx, qy, qz, 0.1)
    for workers in (4 * (os.cpu_count() or 1), 2 ** 31 - 1, 10 ** 30, np.int64(5)):
        assert np.array_equal(grid.count_objects(qx, qy, qz, 0.1, workers=workers), expected)


def test_a_batch_with_workers_raises_naming_the_first_bad_query(worker_grids):
    grid = worker_grids["plain"]
    qx, qy, qz = cloud(3000, seed=41)
    qx[1700] = 2.0
    qz[900] = -1.0
    for workers in (1, 2, 7, -1):
        with pytest.raises(meshsearch.Error, match="query 900"):
            grid.count_objects(qx, qy, qz, 0.1, workers=workers)
        with pytest.raises(meshsearch.Error, match="query 900"):
            grid.close_objects(qx, qy, qz, 0.1, workers=workers)
    objects = np.arange(3000, dtype=np.int64) % len(grid)
    objects[2500] = len(grid) + 5
    for workers in (1, 3, -1):
        with pytest.raises(meshsearch.IndexError, match="query 2500"):
            grid.nearest_object(objects, workers=workers)


# --------------------------------------------------------------------------
# constructor forms
#
# MeshGrid(x, y, z, cellsize) and MeshGrid(pos, cellsize) with pos of shape
# (N, 3) must build the same grid from the same values, whatever the layout
# and dtype of the input: the same box, cells and cell side, the same objects
# in every cell in the same order, and the same answer to every query. The
# reference is built from Python lists, the sequence conversion the
# constructor has always used. float64 C-contiguous input is read in place,
# without a temporary copy; numpy reports its own allocations to tracemalloc,
# so a conversion shows there and a read in place does not.


def grid_signature(grid, seed):
    """Everything a grid answers, for comparing grids built in different ways."""
    lims = np.asarray(grid.lims)
    nx, ny, nz = grid.n_cells
    c = grid.cellsize
    cells = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                centre = [min(lims[a][1], lims[a][0] + (t + 0.5) * c) for a, t in enumerate((i, j, k))]
                cells.append(grid.get_objects_in_cell(*centre).tolist())
    rng = np.random.default_rng(seed)
    q = [lims[a][0] + rng.random(300) * (lims[a][1] - lims[a][0]) for a in range(3)]
    idx = np.arange(0, len(grid), 3)
    n = min(5, len(grid) - 1)
    out = [lims.tobytes(), (nx, ny, nz), c, grid.periodic, len(grid), cells,
           grid.nearest_object(*q).tolist(), grid.nearest_objects(n, *q).tolist(),
           grid.nearest_object(idx).tolist(), grid.nearest_objects(n, idx).tolist()]
    for rmax, rmin in ((1.5 * c, 0.0), (4.0 * c, 0.5 * c)):
        for b in (grid.close_objects(*q, rmax, rmin), grid.close_objects(idx, rmax, rmin)):
            out += [b.indices.tolist(), b.offsets.tolist()]
        out += [grid.count_objects(*q, rmax, rmin).tolist(), grid.count_objects(idx, rmax, rmin).tolist()]
    return out


def ctor_points(n=1500, seed=50):
    """Uniform points with a few coincident ones and both signed zeros, which
    decide the sign of a fitted box edge bit for bit."""
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 3))
    pos[: n // 10] = pos[n // 10: 2 * (n // 10)]
    pos[5:15] = 0.0
    pos[5:15:2] = -0.0
    return pos


CTOR_OPTIONS = [
    ("fitted", 0.12, {}),
    ("box", 0.2, {"limits": [[-0.5, 1.5], [-0.25, 1.0], [-0.1, 1.2]]}),
    ("periodic", 0.15, {"limits": BOX, "periodic": True}),
    ("mixed", 0.1, {"limits": BOX, "periodic": (True, False, True)}),
]


@pytest.mark.parametrize("name, cellsize, kwargs", CTOR_OPTIONS, ids=[o[0] for o in CTOR_OPTIONS])
def test_every_input_layout_builds_the_same_grid(name, cellsize, kwargs):
    pos = ctor_points()
    if name in ("periodic", "mixed"):
        pos = np.clip(pos, 0.0, np.nextafter(1.0, 0.0))
    x, y, z = (pos[:, a].copy() for a in range(3))
    expected = grid_signature(meshsearch.MeshGrid(x.tolist(), y.tolist(), z.tolist(), cellsize, **kwargs), 1)

    fortran = np.asfortranarray(pos)
    wide = np.zeros((len(pos), 5))
    wide[:, 1:4] = pos
    readonly = pos.copy()
    readonly.flags.writeable = False
    big_endian = pos.astype(">f8")
    columns = {
        "float64 copies": (x, y, z),
        "columns of (N, 3)": (pos[:, 0], pos[:, 1], pos[:, 2]),
        "columns of a Fortran (N, 3)": (fortran[:, 0], fortran[:, 1], fortran[:, 2]),
        "read-only": (readonly[:, 0].copy(), readonly[:, 1], readonly[:, 2]),
        "big-endian": (big_endian[:, 0], big_endian[:, 1], big_endian[:, 2]),
        "lists": (x.tolist(), y.tolist(), z.tolist()),
        "tuples": (tuple(x), tuple(y), tuple(z)),
    }
    rows = {
        "C (N, 3)": pos,
        "Fortran (N, 3)": fortran,
        "strided rows": pos.repeat(2, axis=0)[::2],
        "a slice of wider rows": wide[:, 1:4],
        "read-only": readonly,
        "big-endian": big_endian,
        "list of triples": pos.tolist(),
        "object array": pos.astype(object),
    }
    for label, xyz in columns.items():
        assert grid_signature(meshsearch.MeshGrid(*xyz, cellsize, **kwargs), 1) == expected, label
    for label, p in rows.items():
        assert grid_signature(meshsearch.MeshGrid(p, cellsize, **kwargs), 1) == expected, label
    assert grid_signature(meshsearch.MeshGrid(pos=pos, cellsize=cellsize, **kwargs), 1) == expected


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int64, np.int32, np.uint8, np.bool_])
def test_other_dtypes_build_the_grid_of_their_values(dtype):
    """Converted as the sequence conversion converts each element."""
    rng = np.random.default_rng(51)
    pos = (rng.random((800, 3)) * 20).astype(dtype)
    limits = [[0.0, 20.0]] * 3
    lists = [pos[:, a].tolist() for a in range(3)]
    expected = grid_signature(meshsearch.MeshGrid(*lists, 2.5, limits=limits), 2)
    assert grid_signature(meshsearch.MeshGrid(pos, 2.5, limits=limits), 2) == expected
    assert grid_signature(meshsearch.MeshGrid(pos[:, 0], pos[:, 1], pos[:, 2], 2.5, limits=limits), 2) == expected
    columns = [np.ascontiguousarray(pos[:, a]) for a in range(3)]
    assert grid_signature(meshsearch.MeshGrid(*columns, 2.5, limits=limits), 2) == expected


def test_the_grid_keeps_no_reference_to_the_input():
    pos = ctor_points(600, seed=52)
    x, y, z = (pos[:, a].copy() for a in range(3))
    grids = [meshsearch.MeshGrid(pos, 0.15), meshsearch.MeshGrid(x, y, z, 0.15)]
    expected = [grid_signature(g, 3) for g in grids]
    refcounts = [sys.getrefcount(a) for a in (pos, x, y, z)]
    for a in (pos, x, y, z):
        a[...] = np.nan
    del a
    assert [grid_signature(g, 3) for g in grids] == expected
    assert [sys.getrefcount(a) for a in (pos, x, y, z)] == refcounts
    assert expected[0] == expected[1]


def numpy_peak(build):
    import tracemalloc
    tracemalloc.start()
    try:
        build()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_float64_contiguous_input_is_read_without_a_copy():
    pos = ctor_points(200_000, seed=53)
    x, y, z = (pos[:, a].copy() for a in range(3))
    size = pos.shape[0] * 8
    assert numpy_peak(lambda: meshsearch.MeshGrid(pos, 0.02)) < size // 10
    assert numpy_peak(lambda: meshsearch.MeshGrid(x, y, z, 0.02)) < size // 10
    # The controls: a layout that must be converted shows its copy.
    assert numpy_peak(lambda: meshsearch.MeshGrid(np.asfortranarray(pos), 0.02)) >= 3 * size
    assert numpy_peak(lambda: meshsearch.MeshGrid(pos[:, 0], y, z, 0.02)) >= size


def test_pos_errors_are_those_of_separate_arrays():
    pos = ctor_points(300, seed=54)
    for axis in range(3):
        for bad in (NAN, INF, -INF):
            p = pos.copy()
            p[17, axis] = bad
            p[40, 2 - axis] = bad
            for kwargs in ({}, {"limits": BOX}):
                with pytest.raises(meshsearch.Error) as rows:
                    meshsearch.MeshGrid(p, 0.2, **kwargs)
                with pytest.raises(meshsearch.Error) as columns:
                    meshsearch.MeshGrid(p[:, 0], p[:, 1], p[:, 2], 0.2, **kwargs)
                assert str(rows.value) == str(columns.value)
                assert f"coordinate {17 if axis <= 2 - axis else 40} is not finite" in str(rows.value)
        p = pos.copy()
        p[9, axis] = 1.5
        with pytest.raises(meshsearch.Error, match="outside the limits") as rows:
            meshsearch.MeshGrid(p, 0.2, limits=BOX)
        with pytest.raises(meshsearch.Error) as columns:
            meshsearch.MeshGrid(*p.T, 0.2, limits=BOX)
        assert str(rows.value) == str(columns.value)
    for cellsize in (0.0, -1.0, NAN, INF):
        raises_plain_error(lambda: meshsearch.MeshGrid(pos, cellsize))
    raises_plain_error(lambda: meshsearch.MeshGrid(pos, 0.2, periodic=True))
    raises_plain_error(lambda: meshsearch.MeshGrid(pos, 0.2, limits=[[0, 1], [0, 1]]))
    raises_plain_error(lambda: meshsearch.MeshGrid(np.empty((0, 3)), 0.2))
    with pytest.raises(TypeError):
        meshsearch.MeshGrid(pos, 0.2, periodic="yes")
    with pytest.raises(TypeError):
        meshsearch.MeshGrid(pos, 0.2, BOX)
    with pytest.raises(TypeError):
        meshsearch.MeshGrid(pos, "0.2")


def test_no_points_with_limits_build_an_empty_grid():
    for empty in (np.empty((0, 3)), []):
        grid = meshsearch.MeshGrid(empty, 0.2, limits=BOX)
        assert len(grid) == 0
        assert grid.n_cells == meshsearch.MeshGrid([], [], [], 0.2, limits=BOX).n_cells


@pytest.mark.parametrize("bad", [
    np.zeros(30), np.zeros((10, 2)), np.zeros((10, 4)), np.zeros((3, 10)), np.zeros((10, 3, 1)),
    np.float64(1.0), 1.0, [[1.0, 2.0]] * 4, [1, 2, 3], np.zeros((10, 2), dtype=np.int32),
], ids=lambda b: type(b).__name__ + str(getattr(b, "shape", "")))
def test_pos_of_real_numbers_not_of_shape_n_by_3_raises_error(bad):
    """A wrong shape is meshsearch.Error, as in C++ and in the batch forms,
    and not a TypeError."""
    with pytest.raises(meshsearch.Error) as raised:
        meshsearch.MeshGrid(bad, 0.2, limits=BOX)
    assert not isinstance(raised.value, TypeError)


@pytest.mark.parametrize("bad", [
    None, "abc", [[1.0, 2.0, 3.0], [1.0, 2.0]],
    np.array([["1", "2", "3"]]), np.zeros((4, 3), dtype="datetime64[s]"),
], ids=lambda b: type(b).__name__ + str(getattr(b, "shape", "")))
def test_pos_that_is_not_real_numbers_raises_type_error(bad):
    with pytest.raises(TypeError):
        meshsearch.MeshGrid(bad, 0.2, limits=BOX)


@pytest.mark.parametrize("bad", [
    np.zeros((10, 1)), np.zeros((10, 3)), np.float64(1.0), 1.0, [[1.0, 2.0]] * 3,
    np.zeros((2, 2), dtype=np.int64),
], ids=lambda b: type(b).__name__ + str(getattr(b, "shape", "")))
def test_separate_arrays_of_real_numbers_not_one_dimensional_raise_error(bad):
    """A wrong shape is meshsearch.Error, whatever the argument and its type."""
    ok = np.zeros(10)
    for args in ((bad, ok, ok), (ok, bad, ok), (ok, ok, bad)):
        with pytest.raises(meshsearch.Error) as raised:
            meshsearch.MeshGrid(*args, 0.2, limits=BOX)
        assert not isinstance(raised.value, TypeError)


@pytest.mark.parametrize("bad", [
    None, "abc", ["1"] * 10, np.arange(10).astype("datetime64[s]"), (v for v in range(10)),
], ids=lambda b: type(b).__name__ + str(getattr(b, "shape", "")))
def test_separate_arrays_that_are_not_real_numbers_raise_type_error(bad):
    """The cases the sequence conversion refuses, as they always were.
    Complex numbers are not among them: they are converted, with numpy's
    ComplexWarning, as they always were."""
    ok = np.zeros(10)
    with pytest.raises(TypeError):
        meshsearch.MeshGrid(bad, ok, ok, 0.2, limits=BOX)
    with pytest.raises(TypeError):
        meshsearch.MeshGrid(ok, ok, bad, 0.2, limits=BOX)


def test_separate_arrays_of_different_lengths_raise_whatever_their_layout():
    pos = ctor_points(100, seed=55)
    with pytest.raises(meshsearch.Error, match="differ in length"):
        meshsearch.MeshGrid(pos[:, 0], pos[:99, 1], pos[:, 2], 0.2)
    with pytest.raises(meshsearch.Error, match="differ in length"):
        meshsearch.MeshGrid(pos[:, 0].tolist(), pos[:, 1].astype(np.float32), np.zeros(3), 0.2)
