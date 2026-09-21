// Tests for meshsearch::MeshGrid. They check the guarantees stated in
// include/meshsearch/MeshGrid.h, not the implementation behind them, so a
// rewrite of the internals should leave this file passing unchanged.
//
// The load-bearing groups are the ones that compare every query against brute
// force over random points. NDEBUG is undefined below so that assert() stays
// live whatever the build type.

#undef NDEBUG

#include "meshsearch/MeshGrid.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

using meshsearch::Error;
using meshsearch::IndexError;
using meshsearch::MeshGrid;

#define ASSERT_THROWS(expr)                                 \
  do {                                                      \
    bool thrown = false;                                    \
    try { (void)(expr); }                                   \
    catch (const Error&) { thrown = true; }                 \
    assert(thrown);                                         \
  } while (false)

#define ASSERT_OK(expr)                                     \
  do {                                                      \
    try { (void)(expr); }                                   \
    catch (const Error&) { assert(false); }                 \
  } while (false)

#define ASSERT_THROWS_INDEX(expr)                           \
  do {                                                      \
    bool thrown = false;                                    \
    try { (void)(expr); }                                   \
    catch (const IndexError&) { thrown = true; }            \
    assert(thrown);                                         \
  } while (false)

#define ASSERT_THROWS_PLAIN(expr)                           \
  do {                                                      \
    bool thrown = false;                                    \
    try { (void)(expr); }                                   \
    catch (const IndexError&) { assert(false); }            \
    catch (const Error&) { thrown = true; }                 \
    assert(thrown);                                         \
  } while (false)

namespace {

  const double tol = 1.e-9;

  // The brute-force reference: the same points the grid holds, plus which of
  // them are still alive, so an expected answer can be recomputed by scanning
  // all of them. Throughout, a `skip` of -1 means a query about a free point
  // and a skip >= 0 means a query about that object, which is then excluded.
  struct Cloud {
    std::vector<double> X, Y, Z;
    std::vector<char> alive;

    size_t size () const { return X.size(); }

    double dist_sq (const size_t i, const double x, const double y, const double z) const
    {
      const double dx = X[i]-x, dy = Y[i]-y, dz = Z[i]-z;
      return dx*dx+dy*dy+dz*dz;
    }

    std::vector<std::pair<double, unsigned int>> ranked (const double x, const double y, const double z,
                                                         const long skip) const
    {
      std::vector<std::pair<double, unsigned int>> out;
      for (size_t i=0; i<size(); ++i) {
        if (!alive[i] || long(i) == skip) continue;
        out.push_back({dist_sq(i, x, y, z), (unsigned int)i});
      }
      std::sort(out.begin(), out.end());
      return out;
    }

    std::vector<unsigned int> shell (const double x, const double y, const double z,
                                     const double Rmax, const double Rmin, const long skip) const
    {
      std::vector<unsigned int> out;
      const double hi = Rmax*Rmax, lo = Rmin*Rmin;
      for (size_t i=0; i<size(); ++i) {
        if (!alive[i] || long(i) == skip) continue;
        const double d = dist_sq(i, x, y, z);
        if (d <= hi && d >= lo) out.push_back((unsigned int)i);
      }
      return out;
    }
  };

  // n points spread uniformly through the unit box, from a seeded generator so
  // that a failure is reproducible.
  Cloud uniformCloud (const size_t n, std::mt19937& gen)
  {
    std::uniform_real_distribution<double> unit(0., 1.);
    Cloud c;
    for (size_t i=0; i<n; ++i) {
      c.X.push_back(unit(gen));
      c.Y.push_back(unit(gen));
      c.Z.push_back(unit(gen));
    }
    c.alive.assign(n, 1);
    return c;
  }


  // -----------------------------------------------------------------------


  // nearestObject must return an object at the minimum distance. It compares
  // distances, not indices: with duplicate or coincident points several
  // objects tie for nearest and the header promises no particular one, so
  // asserting a specific index would be asserting an implementation detail.
  void checkNearest (const MeshGrid& grid, const Cloud& cloud,
                     const double x, const double y, const double z, const long skip)
  {
    const auto ref = cloud.ranked(x, y, z, skip);

    if (ref.empty()) {
      if (skip < 0) ASSERT_THROWS_PLAIN(grid.nearestObject(x, y, z));
      else ASSERT_THROWS_PLAIN(grid.nearestObject((unsigned int)skip));
      return;
    }

    const unsigned int got = (skip < 0) ? grid.nearestObject(x, y, z)
                                        : grid.nearestObject((unsigned int)skip);
    assert(got != (unsigned int)skip);
    assert(grid.isAlive(got));
    assert(cloud.dist_sq(got, x, y, z) <= ref.front().first+tol);
  }


  // nearestObjects must return exactly N live objects, nearest first, without
  // repeats and without the query object. Compared distance by distance
  // against the brute-force ranking, for the tie reason above.
  void checkNearestN (const MeshGrid& grid, const Cloud& cloud, const unsigned int N,
                      const double x, const double y, const double z, const long skip)
  {
    const auto ref = cloud.ranked(x, y, z, skip);
    if (N > ref.size()) return;

    const auto got = (skip < 0) ? grid.nearestObjects(N, x, y, z)
                                : grid.nearestObjects(N, (unsigned int)skip);
    assert(got.size() == N);

    double previous = -1.;
    for (size_t i=0; i<got.size(); ++i) {
      assert(grid.isAlive(got[i]));
      assert(got[i] != (unsigned int)skip);
      for (size_t j=0; j<i; ++j) assert(got[i] != got[j]);

      const double d = cloud.dist_sq(got[i], x, y, z);
      assert(d >= previous-tol);
      previous = d;
      assert(std::fabs(d-ref[i].first) < tol);
    }
  }


  // closeObjects must return exactly the brute-force shell membership. Here
  // the comparison is on the index sets and is exact: membership is a decision
  // about each object, not a choice between equals, so nothing may be missing
  // or extra. Order is unspecified, so both sides are sorted first.
  void checkShell (const MeshGrid& grid, const Cloud& cloud,
                   const double x, const double y, const double z,
                   const double Rmax, const double Rmin, const long skip)
  {
    auto ref = cloud.shell(x, y, z, Rmax, Rmin, skip);
    auto got = (skip < 0) ? grid.closeObjects(x, y, z, Rmax, Rmin)
                          : grid.closeObjects((unsigned int)skip, Rmax, Rmin);
    std::sort(ref.begin(), ref.end());
    std::sort(got.begin(), got.end());
    assert(got == ref);
  }


  // -----------------------------------------------------------------------


  // The constructor's contract: the accessors report what was built, each side
  // of the box is grown symmetrically to a whole number of cells, a copy
  // answers as the original does, and a default-constructed grid raises or
  // returns empty from every member.
  void testConstruction ()
  {
    const std::vector<double> X = {0.5, 2.5, 9.5};
    const std::vector<double> Y = {0.5, 2.5, 9.5};
    const std::vector<double> Z = {0.5, 2.5, 9.5};

    const MeshGrid grid(X, Y, Z, 1., {{0., 10.}, {0., 10.}, {0., 10.}});

    assert(grid.get_nObjects() == 3);
    assert(grid.get_cellsize() == 1.);
    assert(grid.get_nCells()[0] == 10);
    assert(grid.get_nCells()[1] == 10);
    assert(grid.get_nCells()[2] == 10);

    const auto lims = grid.get_lims();
    assert(lims.size() == 3);
    for (int i=0; i<3; ++i) {
      assert(lims[i].size() == 2);
      assert(std::fabs(lims[i][0]-0.) < tol);
      assert(std::fabs(lims[i][1]-10.) < tol);
    }

    for (unsigned int i=0; i<3; ++i) assert(grid.isAlive(i));
    assert(!grid.isAlive(3));

    const MeshGrid padded(X, Y, Z, 1.);
    const auto plims = padded.get_lims();
    for (int i=0; i<3; ++i) {
      assert(plims[i][0] <= 0.5);
      assert(plims[i][1] >= 9.5);
      const double span = plims[i][1]-plims[i][0];
      const double n = span/padded.get_cellsize();
      assert(std::fabs(n-std::round(n)) < 1.e-6);
      assert(padded.get_nCells()[i] == (unsigned int)std::llround(n));
    }

    const MeshGrid copy = grid;
    assert(copy.get_nObjects() == grid.get_nObjects());
    assert(copy.nearestObject(0.4, 0.4, 0.4) == grid.nearestObject(0.4, 0.4, 0.4));

    MeshGrid empty;
    assert(empty.get_nObjects() == 0);
    assert(empty.get_lims().empty());
    assert(empty.get_cellsize() == 0.);
    assert(!empty.isAlive(0));
    ASSERT_THROWS(empty.nearestObject(0., 0., 0.));
    ASSERT_THROWS_INDEX(empty.nearestObject(0u));
    ASSERT_THROWS(empty.get_Cell(0., 0., 0.));
    ASSERT_THROWS(empty.closeObjects(0., 0., 0., 1.));
    ASSERT_THROWS(empty.addObject(0., 0., 0.));

    const MeshGrid noObjects({}, {}, {}, 1., {{0., 4.}, {0., 4.}, {0., 4.}});
    assert(noObjects.get_nObjects() == 0);
    assert(noObjects.get_ObjectsInCell(1., 1., 1.).empty());
    assert(noObjects.nearestObjects(0, 1., 1., 1.).empty());
    assert(noObjects.closeObjects(1., 1., 1., 2.).empty());
    ASSERT_THROWS(noObjects.nearestObject(1., 1., 1.));

    printf("  construction and accessors: ok\n");
  }


  // get_Cell, get_CellCoords and get_ObjectsInCell over the whole grid, and
  // the box's closed upper end.
  void testCells ()
  {
    const std::vector<double> X = {0.5}, Y = {0.5}, Z = {0.5};
    const MeshGrid grid(X, Y, Z, 1., {{0., 10.}, {0., 4.}, {0., 2.}});

    assert(grid.get_nCells()[0] == 10);
    assert(grid.get_nCells()[1] == 4);
    assert(grid.get_nCells()[2] == 2);

    const auto first = grid.get_CellCoords(0., 0., 0.);
    assert(first[0] == 0 && first[1] == 0 && first[2] == 0);
    assert(grid.get_Cell(0., 0., 0.) == 0);

    // The upper face. The box is closed at both ends, so (10,4,2), the far
    // corner, exactly on the upper face of all three axes, is inside it and
    // must be accepted. Dividing by the cellsize there gives 10, 4 and 2,
    // which are one past the last cell on each axis, so the point belongs to
    // the last cell, (9,3,1), and not to a cell that does not exist.
    const auto last = grid.get_CellCoords(10., 4., 2.);
    assert(last[0] == 9 && last[1] == 3 && last[2] == 1);
    assert(grid.get_Cell(10., 4., 2.) == 9*4*2+3*2+1);

    // One axis at a time, in case an axis is handled separately.
    const auto edgeX = grid.get_CellCoords(10., 0.5, 0.5);
    assert(edgeX[0] == 9);
    const auto edgeY = grid.get_CellCoords(0.5, 4., 0.5);
    assert(edgeY[1] == 3);
    const auto edgeZ = grid.get_CellCoords(0.5, 0.5, 2.);
    assert(edgeZ[2] == 1);

    const auto mid = grid.get_CellCoords(3.7, 2.2, 1.5);
    assert(mid[0] == 3 && mid[1] == 2 && mid[2] == 1);

    // Walk every cell centre: the coordinates must round-trip, and the linear
    // index must be in range and distinct for every cell, so no two cells
    // share a slot and none is unreachable.
    const size_t total = size_t(grid.get_nCells()[0])*grid.get_nCells()[1]*grid.get_nCells()[2];
    std::vector<int> seen(total, 0);
    for (unsigned int ix=0; ix<grid.get_nCells()[0]; ++ix)
      for (unsigned int iy=0; iy<grid.get_nCells()[1]; ++iy)
        for (unsigned int iz=0; iz<grid.get_nCells()[2]; ++iz) {
          const double x = (ix+0.5)*grid.get_cellsize();
          const double y = (iy+0.5)*grid.get_cellsize();
          const double z = (iz+0.5)*grid.get_cellsize();
          const auto coords = grid.get_CellCoords(x, y, z);
          assert(coords[0] == ix && coords[1] == iy && coords[2] == iz);
          const size_t linear = grid.get_Cell(x, y, z);
          assert(linear < total);
          assert(seen[linear] == 0);
          seen[linear] = 1;
        }

    const auto inCell = grid.get_ObjectsInCell(0.9, 0.9, 0.9);
    assert(inCell.size() == 1 && inCell[0] == 0);
    assert(grid.get_ObjectsInCell(5.5, 2.5, 1.5).empty());

    // Same thing through addObject and a query: an object added exactly on the
    // far corner of the box must land in the last cell and be findable there.
    MeshGrid onFace({0.}, {0.}, {0.}, 1., {{0., 3.}, {0., 3.}, {0., 3.}});
    const unsigned int corner = onFace.addObject(3., 3., 3.);
    const auto cell = onFace.get_ObjectsInCell(3., 3., 3.);
    assert(cell.size() == 1 && cell[0] == corner);
    assert(onFace.get_CellCoords(3., 3., 3.)[0] == 2);
    assert(onFace.nearestObject(3., 3., 3.) == corner);
    assert(onFace.nearestObject(2.9, 2.9, 2.9) == corner);

    printf("  cell lookup and box faces: ok\n");
  }


  // Every @throw the header documents actually throws, and the type is the
  // documented one: IndexError for an index that is out of range or removed,
  // plain Error for everything else, including an out-of-range N, which is a
  // count and not an index.
  void testThrows ()
  {
    const std::vector<double> X = {1., 2., 3.};
    const std::vector<double> Y = {1., 2., 3.};
    const std::vector<double> Z = {1., 2., 3.};
    const std::vector<std::vector<double>> lims = {{0., 4.}, {0., 4.}, {0., 4.}};

    ASSERT_THROWS(MeshGrid({1., 2.}, Y, Z, 1., lims));
    ASSERT_THROWS(MeshGrid(X, {1., 2.}, Z, 1., lims));
    ASSERT_THROWS(MeshGrid(X, Y, {1., 2.}, 1., lims));
    ASSERT_THROWS(MeshGrid(X, Y, Z, 0., lims));
    ASSERT_THROWS(MeshGrid(X, Y, Z, -1., lims));
    ASSERT_THROWS(MeshGrid(X, Y, Z, 1., {{0., 4.}, {0., 4.}}));
    ASSERT_THROWS(MeshGrid(X, Y, Z, 1., {{0., 4.}, {0., 4.}, {0.}}));
    ASSERT_THROWS(MeshGrid(X, Y, Z, 1., {{0., 4.}, {0., 4.}, {0., 4.}, {0., 4.}}));
    ASSERT_THROWS(MeshGrid(X, Y, Z, 1., {{2., 4.}, {0., 4.}, {0., 4.}}));
    ASSERT_THROWS(MeshGrid(X, Y, Z, 1., {{0., 4.}, {0., 2.5}, {0., 4.}}));
    ASSERT_THROWS(MeshGrid(X, Y, Z, 1., {{0., 4.}, {0., 4.}, {1.5, 4.}}));
    ASSERT_OK(MeshGrid(X, Y, Z, 1., {{1., 3.}, {1., 3.}, {1., 3.}}));

    MeshGrid grid(X, Y, Z, 1., lims);
    const MeshGrid& cgrid = grid;

    const double out = 5.;
    ASSERT_THROWS(cgrid.nearestObject(out, 1., 1.));
    ASSERT_THROWS(cgrid.nearestObject(1., out, 1.));
    ASSERT_THROWS(cgrid.nearestObject(1., 1., out));
    ASSERT_THROWS(cgrid.nearestObject(-1., 1., 1.));
    ASSERT_THROWS(cgrid.nearestObjects(1, out, 1., 1.));
    ASSERT_THROWS(cgrid.closeObjects(out, 1., 1., 1.));
    ASSERT_THROWS(cgrid.get_Cell(out, 1., 1.));
    ASSERT_THROWS(cgrid.get_CellCoords(out, 1., 1.));
    ASSERT_THROWS(cgrid.get_ObjectsInCell(out, 1., 1.));
    ASSERT_THROWS(grid.addObject(out, 1., 1.));

    ASSERT_THROWS_INDEX(cgrid.nearestObject(3u));
    ASSERT_THROWS_INDEX(cgrid.nearestObjects(1, 3u));
    ASSERT_THROWS_INDEX(cgrid.closeObjects(3u, 1.));
    ASSERT_THROWS_INDEX(grid.removeObject(3u));

    ASSERT_THROWS_PLAIN(cgrid.nearestObjects(4, 1., 1., 1.));
    ASSERT_OK(cgrid.nearestObjects(3, 1., 1., 1.));
    ASSERT_THROWS_PLAIN(cgrid.nearestObjects(3, 0u));
    ASSERT_OK(cgrid.nearestObjects(2, 0u));

    ASSERT_THROWS(cgrid.closeObjects(1., 1., 1., -1.));
    ASSERT_THROWS(cgrid.closeObjects(1., 1., 1., 1., -1.));
    ASSERT_THROWS(cgrid.closeObjects(1., 1., 1., 1., 2.));
    ASSERT_THROWS_PLAIN(cgrid.closeObjects(0u, -1.));
    ASSERT_THROWS_PLAIN(cgrid.closeObjects(0u, 1., -1.));
    ASSERT_THROWS_PLAIN(cgrid.closeObjects(0u, 1., 2.));

    assert(cgrid.closeObjects(1., 1., 1., 100.).size() == 3);

    const double nan_radius = std::numeric_limits<double>::quiet_NaN();
    const double inf_radius = std::numeric_limits<double>::infinity();

    ASSERT_THROWS(cgrid.closeObjects(1., 1., 1., nan_radius));
    ASSERT_THROWS(cgrid.closeObjects(1., 1., 1., 1., nan_radius));
    ASSERT_THROWS(cgrid.closeObjects(1., 1., 1., nan_radius, nan_radius));
    ASSERT_THROWS(cgrid.closeObjects(1., 1., 1., 2., nan_radius));
    ASSERT_THROWS_PLAIN(cgrid.closeObjects(0u, nan_radius));
    ASSERT_THROWS_PLAIN(cgrid.closeObjects(0u, 1., nan_radius));
    ASSERT_THROWS_PLAIN(cgrid.closeObjects(0u, nan_radius, nan_radius));
    ASSERT_THROWS_PLAIN(cgrid.closeObjects(0u, 2., nan_radius));

    ASSERT_THROWS(cgrid.closeObjects(1., 1., 1., -inf_radius));
    ASSERT_THROWS_PLAIN(cgrid.closeObjects(0u, -inf_radius));
    ASSERT_THROWS(cgrid.closeObjects(1., 1., 1., 1., -inf_radius));

    assert(cgrid.closeObjects(1., 1., 1., inf_radius).size() == 3);
    assert(cgrid.closeObjects(1., 1., 1., inf_radius, 0.).size() == 3);
    assert(cgrid.closeObjects(0u, inf_radius).size() == 2);
    assert(cgrid.closeObjects(0u, inf_radius, 1.5).size() == 2);
    assert(cgrid.closeObjects(0u, inf_radius, 2.).size() == 1);
    assert(cgrid.closeObjects(0u, inf_radius, 4.).empty());

    const double nan_coord = nan_radius;
    ASSERT_THROWS(cgrid.nearestObject(nan_coord, 1., 1.));
    ASSERT_THROWS(cgrid.nearestObject(1., nan_coord, 1.));
    ASSERT_THROWS(cgrid.nearestObject(1., 1., nan_coord));
    ASSERT_THROWS(cgrid.nearestObjects(1, nan_coord, 1., 1.));
    ASSERT_THROWS(cgrid.closeObjects(nan_coord, 1., 1., 1.));
    ASSERT_THROWS(cgrid.get_Cell(nan_coord, 1., 1.));
    ASSERT_THROWS(cgrid.get_CellCoords(1., nan_coord, 1.));
    ASSERT_THROWS(cgrid.get_ObjectsInCell(1., 1., nan_coord));
    ASSERT_THROWS(grid.addObject(nan_coord, 1., 1.));
    ASSERT_THROWS(cgrid.nearestObject(inf_radius, 1., 1.));
    ASSERT_THROWS(grid.addObject(1., -inf_radius, 1.));

    MeshGrid single({1.}, {1.}, {1.}, 1., lims);
    ASSERT_THROWS_PLAIN(single.nearestObject(0u));
    assert(single.nearestObject(2., 2., 2.) == 0);
    single.removeObject(0);
    ASSERT_THROWS_PLAIN(single.nearestObject(2., 2., 2.));
    ASSERT_THROWS_INDEX(single.nearestObject(0u));

    bool caught_as_base = false, caught_as_runtime = false, caught_as_std = false;
    try { grid.removeObject(3u); } catch (const Error&) { caught_as_base = true; }
    try { grid.removeObject(3u); } catch (const std::runtime_error&) { caught_as_runtime = true; }
    try { grid.removeObject(3u); } catch (const std::exception&) { caught_as_std = true; }
    assert(caught_as_base && caught_as_runtime && caught_as_std);

    try { grid.removeObject(3u); assert(false); }
    catch (const IndexError& e) { assert(std::string(e.what()).size() > 0); }

    ASSERT_THROWS_INDEX(cgrid.nearestObjects(99, 3u));
    ASSERT_THROWS_INDEX(cgrid.closeObjects(3u, -1.));
    ASSERT_THROWS_INDEX(cgrid.closeObjects(3u, nan_radius));
    ASSERT_THROWS_INDEX(cgrid.closeObjects(3u, 1., 2.));
    ASSERT_THROWS_INDEX(cgrid.nearestObject(std::numeric_limits<unsigned int>::max()));

    MeshGrid removed({1., 2.}, {1., 2.}, {1., 2.}, 1., lims);
    removed.removeObject(1);
    ASSERT_THROWS_INDEX(removed.nearestObjects(50, 1u));
    ASSERT_THROWS_INDEX(removed.closeObjects(1u, -3.));
    ASSERT_THROWS_PLAIN(removed.nearestObject(0u));
    ASSERT_THROWS_PLAIN(removed.nearestObjects(1, 0u));

    printf("  documented throws, and IndexError for a bad index: ok\n");
  }


  // Index stability: removing an object leaves every other index valid and its
  // own permanently invalid, adding issues an index one past the highest ever
  // issued and never reuses a removed one, get_nObjects tracks both, and
  // queries keep agreeing with brute force across the changes.
  void testIndexStability ()
  {
    std::mt19937 gen(20250921u);
    Cloud cloud = uniformCloud(400, gen);
    const std::vector<std::vector<double>> lims = {{0., 1.}, {0., 1.}, {0., 1.}};

    MeshGrid grid(cloud.X, cloud.Y, cloud.Z, 0.12, lims);
    assert(grid.get_nObjects() == 400);

    std::vector<unsigned int> removed;
    for (unsigned int i=7; i<400; i+=11) removed.push_back(i);

    unsigned int expected = 400;
    for (const auto i : removed) {
      assert(grid.isAlive(i));
      grid.removeObject(i);
      cloud.alive[i] = 0;
      expected--;
      assert(grid.get_nObjects() == expected);
      assert(!grid.isAlive(i));
    }

    for (const auto i : removed) {
      ASSERT_THROWS_INDEX(grid.removeObject(i));
      ASSERT_THROWS_INDEX(grid.nearestObject(i));
      ASSERT_THROWS_INDEX(grid.nearestObjects(1, i));
      ASSERT_THROWS_INDEX(grid.closeObjects(i, 0.1));
    }

    for (size_t i=0; i<cloud.size(); ++i) {
      if (!cloud.alive[i]) continue;
      assert(grid.isAlive((unsigned int)i));
      ASSERT_OK(grid.nearestObject((unsigned int)i));
      const auto cell = grid.get_ObjectsInCell(cloud.X[i], cloud.Y[i], cloud.Z[i]);
      assert(std::find(cell.begin(), cell.end(), (unsigned int)i) != cell.end());
    }

    for (const auto i : removed) {
      const auto cell = grid.get_ObjectsInCell(cloud.X[i], cloud.Y[i], cloud.Z[i]);
      assert(std::find(cell.begin(), cell.end(), i) == cell.end());
    }

    std::uniform_real_distribution<double> unit(0., 1.);
    for (int q=0; q<80; ++q) {
      const double x = unit(gen), y = unit(gen), z = unit(gen);
      checkNearest(grid, cloud, x, y, z, -1);
      checkNearestN(grid, cloud, 6, x, y, z, -1);
      checkShell(grid, cloud, x, y, z, 0.2, 0., -1);
    }

    unsigned int highest = 399;
    for (int a=0; a<40; ++a) {
      const double x = unit(gen), y = unit(gen), z = unit(gen);
      const unsigned int index = grid.addObject(x, y, z);
      assert(index > highest);
      highest = index;
      assert(grid.isAlive(index));
      assert(index >= cloud.size());
      while (cloud.size() < index) {
        cloud.X.push_back(0.); cloud.Y.push_back(0.); cloud.Z.push_back(0.); cloud.alive.push_back(0);
      }
      cloud.X.push_back(x); cloud.Y.push_back(y); cloud.Z.push_back(z); cloud.alive.push_back(1);
      expected++;
      assert(grid.get_nObjects() == expected);
      for (const auto r : removed) assert(!grid.isAlive(r));
    }

    for (int q=0; q<80; ++q) {
      const double x = unit(gen), y = unit(gen), z = unit(gen);
      checkNearest(grid, cloud, x, y, z, -1);
      checkNearestN(grid, cloud, 9, x, y, z, -1);
      checkShell(grid, cloud, x, y, z, 0.25, 0.05, -1);
    }
    for (unsigned int i=0; i<cloud.size(); ++i) {
      if (!cloud.alive[i]) continue;
      checkNearest(grid, cloud, cloud.X[i], cloud.Y[i], cloud.Z[i], (long)i);
      checkShell(grid, cloud, cloud.X[i], cloud.Y[i], cloud.Z[i], 0.2, 0., (long)i);
    }

    while (grid.get_nObjects() > 0) {
      for (unsigned int i=0; i<cloud.size(); ++i)
        if (cloud.alive[i]) { grid.removeObject(i); cloud.alive[i] = 0; break; }
    }
    assert(grid.get_nObjects() == 0);
    ASSERT_THROWS(grid.nearestObject(0.5, 0.5, 0.5));
    const unsigned int reborn = grid.addObject(0.5, 0.5, 0.5);
    assert(reborn == highest+1);
    assert(grid.get_nObjects() == 1);
    assert(grid.nearestObject(0.4, 0.4, 0.4) == reborn);

    printf("  index stability across removals and additions: ok\n");
  }


  // Construction with no limits, where the padding proportional to the data
  // extent has nothing to be proportional to: a single point, a wholly
  // coincident cloud, a flat one, and clouds with extreme extent or offset
  // must all still yield a grid of at least one cell per axis that answers
  // correctly.
  void testDegenerateExtents ()
  {
    const MeshGrid single({3.}, {4.}, {5.}, 0.7);
    assert(single.get_nObjects() == 1);
    for (int i=0; i<3; ++i) assert(single.get_nCells()[i] >= 1);
    assert(single.nearestObject(3., 4., 5.) == 0);
    assert(single.closeObjects(3., 4., 5., 0.).size() == 1);

    const MeshGrid coincident({2., 2., 2.}, {2., 2., 2.}, {2., 2., 2.}, 1.);
    assert(coincident.get_nObjects() == 3);
    for (int i=0; i<3; ++i) assert(coincident.get_nCells()[i] >= 1);
    assert(coincident.closeObjects(2., 2., 2., 0.).size() == 3);
    assert(coincident.closeObjects(0u, 0.).size() == 2);
    assert(coincident.nearestObjects(2, 0u).size() == 2);

    const MeshGrid planar({0., 1., 2., 3.}, {0., 1., 2., 3.}, {7., 7., 7., 7.}, 0.5);
    assert(planar.get_nCells()[2] >= 1);
    assert(planar.nearestObject(0.1, 0.1, 7.) == 0);
    assert(planar.closeObjects(0u, 1.5).size() == 1);

    const MeshGrid elongated({0., 1.e6}, {0., 1.}, {0., 1.}, 1000.);
    assert(elongated.nearestObject(1.e6, 1., 1.) == 1);
    assert(elongated.nearestObject(0u) == 1);

    const MeshGrid offset({1.e9, 1.e9+1.}, {0., 0.}, {0., 0.}, 0.5);
    assert(offset.nearestObject(1.e9, 0., 0.) == 0);
    assert(offset.nearestObject(1.e9+1., 0., 0.) == 1);
    assert(offset.nearestObject(0u) == 1);

    for (int i=0; i<3; ++i) {
      const auto lims = single.get_lims();
      const double span = lims[i][1]-lims[i][0];
      assert(std::fabs(span-single.get_nCells()[i]*single.get_cellsize()) < 1.e-9);
    }

    printf("  degenerate and extreme extents: ok\n");
  }

  // The load-bearing group: every query kind against brute force over several
  // thousand random points, at three cell sizes relative to the mean
  // separation. This is what establishes that the geometry is right, rather
  // The Rmax layer bound, swept across the ratio that drives it. A layer L
  // guarantees a minimum distance of L cell sides, so the bound scans layers 0
  // to floor(Rmax/cellsize) and no further. The ratio is what decides how many
  // layers that is, so it is what has to be swept: from a shell far inside one
  // cell to one several cells wide, against brute force at each step.
  void testShellRadiusSweep ()
  {
    const std::vector<std::vector<double>> lims = {{0., 1.}, {0., 1.}, {0., 1.}};
    std::mt19937 gen(99u);
    const Cloud cloud = uniformCloud(1500, gen);

    const double ratios[] = {0.05, 0.1, 0.25, 0.4, 0.5, 0.75, 0.99, 1., 1.01,
                             1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 4., 5.5, 8.};
    const double cellsizes[] = {0.3, 0.12, 0.05};

    std::uniform_real_distribution<double> unit(0., 1.);
    std::uniform_int_distribution<unsigned int> pick(0, 1499);

    for (const double cellsize : cellsizes) {
      const MeshGrid grid(cloud.X, cloud.Y, cloud.Z, cellsize, lims);

      for (const double ratio : ratios) {
        const double Rmax = ratio*cellsize;

        for (int q=0; q<6; ++q) {
          const double x = unit(gen), y = unit(gen), z = unit(gen);
          checkShell(grid, cloud, x, y, z, Rmax, 0., -1);
          checkShell(grid, cloud, x, y, z, Rmax, 0.5*Rmax, -1);

          const unsigned int index = pick(gen);
          checkShell(grid, cloud, cloud.X[index], cloud.Y[index], cloud.Z[index],
                     Rmax, 0., (long)index);
          checkShell(grid, cloud, cloud.X[index], cloud.Y[index], cloud.Z[index],
                     Rmax, 0.5*Rmax, (long)index);
        }
      }
    }

    printf("  shell radius swept across cell sides: ok\n");
  }


  // The tightest configurations the Rmax bound admits: an object sitting in
  // the very last layer the bound scans, at exactly the shell's outer edge.
  // Random data does not produce these, and one layer tighter loses every one
  // of them.
  //
  // Cellsize is 1, so a layer index is a distance in cell sides. Every
  // coordinate below is exact in binary, so the boundary comparisons are not
  // decided by rounding.
  void testTightestScannedLayer ()
  {
    const std::vector<std::vector<double>> lims = {{0., 20.}, {0., 20.}, {0., 20.}};

    auto sorted = [](std::vector<unsigned int> v) {
      std::sort(v.begin(), v.end());
      return v;
    };

    // The query point sits 0.75 into cell 10 at x = 10.75. The object sits at
    // x = 13, the low corner of cell 13: an offset of 3, per-axis gap 3-1 = 2,
    // layer floor(sqrt(4)) = 2. Their distance is 2.25, and that is the
    // closest an object in any layer-2 cell can be to this query point.
    // Rmax = 2.25 makes floor(Rmax/cellsize) = 2, so layer 2 is the last one
    // scanned and the object lies exactly on the outer edge.
    {
      const MeshGrid grid({10.75, 13.}, {10.5, 10.5}, {10.5, 10.5}, 1., lims);
      assert(grid.get_CellCoords(10.75, 10.5, 10.5)[0] == 10);
      assert(grid.get_CellCoords(13., 10.5, 10.5)[0] == 13);
      assert(sorted(grid.closeObjects(10.75, 10.5, 10.5, 2.25)) == std::vector<unsigned int>({0, 1}));
      assert(sorted(grid.closeObjects(0u, 2.25)) == std::vector<unsigned int>({1}));
      assert(grid.closeObjects(0u, 2.25, 2.25).size() == 1);
      // a hair inside the edge and the object is correctly out of the shell
      assert(grid.closeObjects(0u, 2.2421875).empty());
    }

    // The same at layer 4: offset 5, gap 4, distance 4.25 from x = 10.75 to
    // x = 15, with Rmax = 4.25 so floor(Rmax/cellsize) = 4.
    {
      const MeshGrid grid({10.75, 15.}, {10.5, 10.5}, {10.5, 10.5}, 1., lims);
      assert(sorted(grid.closeObjects(0u, 4.25)) == std::vector<unsigned int>({1}));
      assert(grid.closeObjects(0u, 4.25, 4.25).size() == 1);
    }

    // Off axis, where all three gaps contribute: offset (3,3,3), gaps (2,2,2),
    // layer floor(sqrt(12)) = 3, distance 2.25*sqrt(3) = 3.897. Rmax = 3.9375
    // keeps floor(Rmax/cellsize) = 3, so layer 3 is again the last scanned.
    {
      const MeshGrid grid({10.75, 13.}, {10.75, 13.}, {10.75, 13.}, 1., lims);
      const double d = std::sqrt(3.*2.25*2.25);
      assert(d > 3. && d < 3.9375);
      assert(sorted(grid.closeObjects(0u, 3.9375)) == std::vector<unsigned int>({1}));
    }

    printf("  tightest layer the Rmax bound admits: ok\n");
  }


  // than merely self-consistent.
  void testBruteForce ()
  {
    const std::vector<std::vector<double>> lims = {{0., 1.}, {0., 1.}, {0., 1.}};
    const size_t n = 3000;
    const double mean_separation = 1./std::cbrt(double(n));

    const double factors[] = {4., 1., 0.3};
    const char* names[] = {"dense", "matched", "sparse"};

    for (int r=0; r<3; ++r) {
      std::mt19937 gen(4242u+r);
      const Cloud cloud = uniformCloud(n, gen);
      const double cellsize = factors[r]*mean_separation;

      const MeshGrid grid(cloud.X, cloud.Y, cloud.Z, cellsize, lims);
      assert(grid.get_nObjects() == n);

      std::uniform_real_distribution<double> unit(0., 1.);
      std::uniform_int_distribution<unsigned int> pick(0, (unsigned int)n-1);

      for (int q=0; q<120; ++q) {
        const double x = unit(gen), y = unit(gen), z = unit(gen);
        checkNearest(grid, cloud, x, y, z, -1);
        checkNearestN(grid, cloud, 1, x, y, z, -1);
        checkNearestN(grid, cloud, 5, x, y, z, -1);
        checkNearestN(grid, cloud, 17, x, y, z, -1);
        assert(grid.nearestObjects(0, x, y, z).empty());

        const unsigned int index = pick(gen);
        checkNearest(grid, cloud, cloud.X[index], cloud.Y[index], cloud.Z[index], (long)index);
        checkNearestN(grid, cloud, 1, cloud.X[index], cloud.Y[index], cloud.Z[index], (long)index);
        checkNearestN(grid, cloud, 5, cloud.X[index], cloud.Y[index], cloud.Z[index], (long)index);
        checkNearestN(grid, cloud, 17, cloud.X[index], cloud.Y[index], cloud.Z[index], (long)index);
        assert(grid.nearestObjects(0, index).empty());
      }

      const double radii[][2] = {{0.03, 0.}, {0.1, 0.}, {0.1, 0.05}, {0.3, 0.2}, {0.05, 0.05}};
      for (int q=0; q<40; ++q) {
        const double x = unit(gen), y = unit(gen), z = unit(gen);
        const unsigned int index = pick(gen);
        for (const auto& rr : radii) {
          checkShell(grid, cloud, x, y, z, rr[0], rr[1], -1);
          checkShell(grid, cloud, cloud.X[index], cloud.Y[index], cloud.Z[index], rr[0], rr[1], (long)index);
        }
      }

      const unsigned int all = grid.nearestObjects((unsigned int)n, 0.5, 0.5, 0.5).size();
      assert(all == n);

      printf("  brute force, %s regime (cellsize %.4f, mean separation %.4f): ok\n",
             names[r], cellsize, mean_separation);
    }
  }


  // closeObjects at large Rmin, where the scan may start past the innermost
  // mask layers. Random shells first, then the one configuration that random
  // data does not reach (explained at the point it is built).
  void testWideShells ()
  {
    const std::vector<std::vector<double>> lims = {{0., 1.}, {0., 1.}, {0., 1.}};
    std::mt19937 gen(777u);
    const Cloud cloud = uniformCloud(2000, gen);

    const double cellsize = 0.05;
    const MeshGrid grid(cloud.X, cloud.Y, cloud.Z, cellsize, lims);

    std::uniform_real_distribution<double> unit(0., 1.);
    std::uniform_int_distribution<unsigned int> pick(0, 1999);

    for (int k=1; k<=14; ++k) {
      const double Rmin = k*cellsize;
      const double Rmax = Rmin+0.5*cellsize;
      for (int q=0; q<12; ++q) {
        const double x = unit(gen), y = unit(gen), z = unit(gen);
        checkShell(grid, cloud, x, y, z, Rmax, Rmin, -1);
        const unsigned int index = pick(gen);
        checkShell(grid, cloud, cloud.X[index], cloud.Y[index], cloud.Z[index], Rmax, Rmin, (long)index);
      }
    }

    // The layer-skip case, built by hand because random shells do not produce
    // it. A shell query starts its scan some layers out, and how many it may
    // skip has to be derived from Rmin. A mask layer L records a cell's
    // *minimum* distance from the target cell and says nothing about its
    // maximum, so a bound that skips too eagerly can skip a cell that still
    // reaches into the shell.
    //
    // Cellsize is 1. The query point (10,10,10) sits exactly on the low corner
    // of cell (10,10,10). The second object sits just inside cell (16,16,16),
    // a hair short of its far corner at (17,17,17); the 1e-9 keeps it in that
    // cell rather than the next one.
    //
    // That is an offset of (6,6,6) from the target cell, so the per-axis gaps
    // are 6-1 = 5 and the layer is floor(sqrt(75)) = 8: the mask guarantees
    // only that the cell is at least 8.660 cellsize away, while its far corner
    // is at 7*sqrt(3) = 12.124. With Rmin = 12 a bound skipping every layer
    // below floor(12/1) - 3 = 9 skips layer 8, and this object with it. The
    // bound the library uses subtracts 4 instead, so the scan starts at layer
    // 8 and finds it.
    //
    // The failure needs the query point in one corner of its cell and the
    // object in the opposite corner of a diagonally offset cell, which is why
    // the random shells above never hit it.
    Cloud corner;
    corner.X = {10., 17.-1.e-9};
    corner.Y = {10., 17.-1.e-9};
    corner.Z = {10., 17.-1.e-9};
    corner.alive.assign(2, 1);

    const MeshGrid corner_grid(corner.X, corner.Y, corner.Z, 1.,
                               {{0., 40.}, {0., 40.}, {0., 40.}});

    const double d = std::sqrt(corner.dist_sq(1, 10., 10., 10.));
    assert(d > 12. && d < 12.2);
    checkShell(corner_grid, corner, 10., 10., 10., 12.2, 12., -1);
    checkShell(corner_grid, corner, 10., 10., 10., 12.2, 12., 0);
    assert(corner_grid.closeObjects(0u, 12.2, 12.).size() == 1);

    // Then sweep the same two points across shells from Rmin 2 to 20 and
    // widths up to 11.5, so every layer-skip arithmetic a shell can produce on
    // this configuration is exercised, not only the Rmin = 12 case above.
    for (int k=2; k<=20; ++k) {
      const double Rmin = double(k);
      for (int step=0; step<24; ++step) {
        const double delta = 0.5*step;
        checkShell(corner_grid, corner, 10., 10., 10., Rmin+delta, Rmin, -1);
      }
    }

    printf("  wide shells at large Rmin: ok\n");
  }


  // closeObjects' bounds are inclusive at both ends, Rmin <= d <= Rmax, on
  // hand-placed points whose distances are exact in binary so the boundary is
  // unambiguous; and the index overload excludes the query object itself but
  // not other objects coincident with it.
  void testRminBoundary ()
  {
    const std::vector<double> X = {5., 6., 7., 8., 5., 5.};
    const std::vector<double> Y = {5., 5., 5., 5., 7., 8.};
    const std::vector<double> Z = {5., 5., 5., 5., 5., 5.};
    const std::vector<std::vector<double>> lims = {{0., 16.}, {0., 16.}, {0., 16.}};

    const MeshGrid grid(X, Y, Z, 1., lims);

    auto sorted = [](std::vector<unsigned int> v) {
      std::sort(v.begin(), v.end());
      return v;
    };

    assert(sorted(grid.closeObjects(5., 5., 5., 2., 2.)) == std::vector<unsigned int>({2, 4}));
    assert(sorted(grid.closeObjects(0u, 2., 2.)) == std::vector<unsigned int>({2, 4}));
    assert(sorted(grid.closeObjects(0u, 3., 2.)) == std::vector<unsigned int>({2, 3, 4, 5}));
    assert(sorted(grid.closeObjects(0u, 3., 3.)) == std::vector<unsigned int>({3, 5}));

    assert(sorted(grid.closeObjects(0u, 0., 0.)).empty());
    assert(sorted(grid.closeObjects(5., 5., 5., 0., 0.)) == std::vector<unsigned int>({0}));

    const std::vector<double> dX = {2., 2., 2., 4.};
    const std::vector<double> dY = {2., 2., 2., 2.};
    const std::vector<double> dZ = {2., 2., 2., 2.};
    const MeshGrid dup(dX, dY, dZ, 1., lims);

    assert(sorted(dup.closeObjects(2., 2., 2., 0.)) == std::vector<unsigned int>({0, 1, 2}));
    assert(sorted(dup.closeObjects(0u, 0.)) == std::vector<unsigned int>({1, 2}));
    assert(sorted(dup.closeObjects(0u, 0., 0.)) == std::vector<unsigned int>({1, 2}));
    assert(dup.nearestObject(0u) == 1 || dup.nearestObject(0u) == 2);
    assert(dup.nearestObjects(2, 0u).size() == 2);
    assert(sorted(dup.nearestObjects(2, 0u)) == std::vector<unsigned int>({1, 2}));
    assert(dup.nearestObjects(3, 0u)[2] == 3);

    printf("  Rmin boundary and coincident points: ok\n");
  }


  // The same queries against a cloud where a third of the points duplicate an
  // earlier one exactly: ties must not cost or duplicate a result.
  void testDuplicates ()
  {
    const std::vector<std::vector<double>> lims = {{0., 1.}, {0., 1.}, {0., 1.}};
    std::mt19937 gen(31337u);
    std::uniform_real_distribution<double> unit(0., 1.);
    std::uniform_int_distribution<int> coin(0, 2);

    Cloud cloud;
    for (size_t i=0; i<1500; ++i) {
      if (i > 0 && coin(gen) == 0) {
        const size_t source = i/2;
        cloud.X.push_back(cloud.X[source]);
        cloud.Y.push_back(cloud.Y[source]);
        cloud.Z.push_back(cloud.Z[source]);
      }
      else {
        cloud.X.push_back(unit(gen));
        cloud.Y.push_back(unit(gen));
        cloud.Z.push_back(unit(gen));
      }
    }
    cloud.alive.assign(cloud.size(), 1);

    const double cellsizes[] = {0.4, 0.09, 0.02};
    for (const double cellsize : cellsizes) {
      const MeshGrid grid(cloud.X, cloud.Y, cloud.Z, cellsize, lims);
      std::uniform_int_distribution<unsigned int> pick(0, (unsigned int)cloud.size()-1);

      for (int q=0; q<60; ++q) {
        const double x = unit(gen), y = unit(gen), z = unit(gen);
        checkNearest(grid, cloud, x, y, z, -1);
        checkNearestN(grid, cloud, 4, x, y, z, -1);
        checkNearestN(grid, cloud, 12, x, y, z, -1);
        checkShell(grid, cloud, x, y, z, 0.1, 0., -1);

        const unsigned int index = pick(gen);
        const double px = cloud.X[index], py = cloud.Y[index], pz = cloud.Z[index];
        checkNearest(grid, cloud, px, py, pz, (long)index);
        checkNearestN(grid, cloud, 4, px, py, pz, (long)index);
        checkNearestN(grid, cloud, 12, px, py, pz, (long)index);
        checkShell(grid, cloud, px, py, pz, 0.1, 0., (long)index);
        checkShell(grid, cloud, px, py, pz, 0.15, 0.1, (long)index);

        const double zero_shell = grid.closeObjects(index, 0.).size();
        const double zero_ref = cloud.shell(px, py, pz, 0., 0., (long)index).size();
        assert(zero_shell == zero_ref);
      }
      printf("  duplicates and coincident points (cellsize %.2f): ok\n", cellsize);
    }
  }

}


int main ()
{
  printf("meshsearch::MeshGrid\n");

  testConstruction();
  testCells();
  testThrows();
  testIndexStability();
  testRminBoundary();
  testDuplicates();
  testDegenerateExtents();
  testWideShells();
  testShellRadiusSweep();
  testTightestScannedLayer();
  testBruteForce();

  printf("all tests passed\n");

  return 0;
}
