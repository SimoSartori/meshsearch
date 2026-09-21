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
    return {name: meshsearch.MeshGrid(X, Y, Z, size, BOX) for name, size in REGIMES}


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
        "get_cell",
        "get_cell_coords",
        "get_objects_in_cell",
        "is_alive",
        "lims",
        "n_cells",
        "nearest_object",
        "nearest_objects",
        "remove_object",
    }
    for absent in ("nearestObject", "closerObject", "get_lims", "get_nCells",
                   "get_cellsize", "get_nObjects", "n_objects", "isAlive",
                   "removeObject", "addObject"):
        assert not hasattr(meshsearch.MeshGrid, absent)


def test_len_is_the_live_count(points):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, MPS, BOX)
    assert len(grid) == len(X)
    grid.remove_object(0)
    assert len(grid) == len(X) - 1
    grid.add_object(0.5, 0.5, 0.5)
    assert len(grid) == len(X)


def test_properties(points):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, 0.25, BOX)

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
    grid = meshsearch.MeshGrid(X, Y, Z, 0.25, BOX)
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
    from_arrays = meshsearch.MeshGrid(X, Y, Z, MPS, BOX)
    from_lists = meshsearch.MeshGrid(X.tolist(), Y.tolist(), Z.tolist(), MPS, BOX)
    assert len(from_arrays) == len(from_lists)
    assert from_arrays.nearest_object(0.5, 0.5, 0.5) == from_lists.nearest_object(0.5, 0.5, 0.5)


def test_mismatched_input_lengths_raise(points):
    X, Y, Z = points
    with pytest.raises(meshsearch.Error):
        meshsearch.MeshGrid(X[:-1], Y, Z, MPS, BOX)
    with pytest.raises(meshsearch.Error):
        meshsearch.MeshGrid(X, Y[:-1], Z, MPS, BOX)
    with pytest.raises(meshsearch.Error):
        meshsearch.MeshGrid(X, Y, Z[:-1], MPS, BOX)


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
    grid = meshsearch.MeshGrid(X, Y, Z, MPS, BOX)
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
    grid = meshsearch.MeshGrid(X, Y, Z, MPS, BOX)
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
    grid = meshsearch.MeshGrid(X, Y, Z, MPS, BOX)
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
    grid = meshsearch.MeshGrid(X, Y, Z, MPS, BOX)
    grid.remove_object(11)
    with pytest.raises(meshsearch.IndexError):
        call(grid, 11)


def test_add_object_never_reuses_an_index(points):
    X, Y, Z = points
    grid = meshsearch.MeshGrid(X, Y, Z, MPS, BOX)

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
                               [[0.0, 10.0], [0.0, 4.0], [0.0, 2.0]])
    assert grid.n_cells == (10, 4, 2)
    assert grid.get_cell_coords(10.0, 4.0, 2.0) == (9, 3, 1)
    assert grid.get_cell_coords(0.0, 0.0, 0.0) == (0, 0, 0)

    added = grid.add_object(10.0, 4.0, 2.0)
    assert grid.get_objects_in_cell(10.0, 4.0, 2.0).tolist() == [added]
    assert grid.nearest_object(10.0, 4.0, 2.0) == added


def test_rmin_and_rmax_are_both_inclusive():
    # distances 1, 2 and 3 along x and 2, 3 along y from the first object, all
    # exact in binary so the shell boundary is unambiguous
    X = [5.0, 6.0, 7.0, 8.0, 5.0, 5.0]
    Y = [5.0, 5.0, 5.0, 5.0, 7.0, 8.0]
    Z = [5.0] * 6
    grid = meshsearch.MeshGrid(X, Y, Z, 1.0, [[0.0, 16.0]] * 3)

    assert set(grid.close_objects(5.0, 5.0, 5.0, 2.0, 2.0).tolist()) == {2, 4}
    assert set(grid.close_objects(0, 2.0, 2.0).tolist()) == {2, 4}
    assert set(grid.close_objects(0, 3.0, 2.0).tolist()) == {2, 3, 4, 5}
    assert set(grid.close_objects(0, 3.0, 3.0).tolist()) == {3, 5}


def test_coincident_points():
    X = [2.0, 2.0, 2.0, 4.0]
    grid = meshsearch.MeshGrid(X, [2.0] * 3 + [2.0], [2.0] * 4, 1.0,
                               [[0.0, 16.0]] * 3)

    # a point coincident with the query object is a neighbour at distance 0,
    # the query object itself is not
    assert set(grid.close_objects(2.0, 2.0, 2.0, 0.0).tolist()) == {0, 1, 2}
    assert set(grid.close_objects(0, 0.0).tolist()) == {1, 2}
    assert grid.nearest_object(0) in (1, 2)
    assert set(grid.nearest_objects(2, 0).tolist()) == {1, 2}
    assert grid.nearest_objects(3, 0).tolist()[2] == 3


@pytest.mark.parametrize("cellsize", [0.3, 0.12, 0.05])
def test_close_objects_across_the_radius_to_cellsize_ratio(cellsize):
    """The Rmax bound scans layers 0 to floor(Rmax/cellsize), so that ratio is
    what decides how many layers are scanned and what has to be swept."""
    X, Y, Z = cloud(1500, seed=99)
    alive = np.ones(1500, dtype=bool)
    grid = meshsearch.MeshGrid(X, Y, Z, cellsize, BOX)
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
                               [[0.0, 20.0]] * 3)
    assert grid.get_cell_coords(10.75, 10.5, 10.5)[0] == 10
    assert grid.get_cell_coords(13.0, 10.5, 10.5)[0] == 13
    assert grid.close_objects(0, 2.25).tolist() == [1]
    assert grid.close_objects(0, 2.25, 2.25).tolist() == [1]
    assert grid.close_objects(0, 2.2421875).size == 0


def test_a_wide_shell_at_large_rmin():
    # the configuration random data does not reach: the query point on the low
    # corner of its cell and the object just inside the far corner of the cell
    # six cells away on each axis, at 7*sqrt(3) = 12.124 cellsize
    grid = meshsearch.MeshGrid([10.0, 17.0 - 1e-9], [10.0, 17.0 - 1e-9],
                               [10.0, 17.0 - 1e-9], 1.0, [[0.0, 40.0]] * 3)
    assert grid.close_objects(10.0, 10.0, 10.0, 12.2, 12.0).tolist() == [1]
    assert grid.close_objects(0, 12.2, 12.0).tolist() == [1]
