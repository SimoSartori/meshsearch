"""Tests for the Python bindings.

They cover the guarantees the C++ suite covers -- checked the same way, against
brute force over random points -- plus what only exists at the language
boundary: the returned dtypes, the properties and len(), and the mapping of the
two C++ exception types onto Python ones.
"""

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
