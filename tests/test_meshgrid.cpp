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
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

using meshsearch::Error;
using meshsearch::IndexError;
using meshsearch::MeshGrid;

namespace {

  // The box of a 3x2 vector of vectors, as options.
  MeshGrid::Options boxOf (const std::vector<std::vector<double>>& lims)
  {
    assert(lims.size() == 3);
    MeshGrid::Options options;
    std::array<std::array<double,2>,3> limits;
    for (int a=0; a<3; ++a) {
      assert(lims[a].size() == 2);
      limits[a] = {lims[a][0], lims[a][1]};
    }
    options.limits = limits;
    return options;
  }

}

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
  // repeats and without the query object. Unlike nearestObject, the header
  // fixes the order of ties (ascending index, also for the last place kept),
  // so the result must equal the first N of the brute-force ranking by
  // (distance, index) index for index. The reference computes distances as
  // the library does, with the same contraction setting, so equal distances
  // are equal in both.
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
      assert(got[i] == ref[i].second);
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

    const MeshGrid grid(X, Y, Z, 1., boxOf({{0., 10.}, {0., 10.}, {0., 10.}}));

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
    // get_lims returns a fixed-size array, so a grid with no box reports
    // zeros, and get_nCells tells such a grid apart.
    assert((empty.get_lims() == std::array<std::array<double,2>,3>{}));
    assert(empty.get_cellsize() == 0.);
    assert(!empty.isAlive(0));
    ASSERT_THROWS(empty.nearestObject(0., 0., 0.));
    ASSERT_THROWS_INDEX(empty.nearestObject(0u));
    ASSERT_THROWS(empty.get_cell(0., 0., 0.));
    ASSERT_THROWS(empty.closeObjects(0., 0., 0., 1.));
    ASSERT_THROWS(empty.addObject(0., 0., 0.));

    const MeshGrid noObjects({}, {}, {}, 1., boxOf({{0., 4.}, {0., 4.}, {0., 4.}}));
    assert(noObjects.get_nObjects() == 0);
    assert(noObjects.get_objectsInCell(1., 1., 1.).empty());
    assert(noObjects.nearestObjects(0, 1., 1., 1.).empty());
    assert(noObjects.closeObjects(1., 1., 1., 2.).empty());
    ASSERT_THROWS(noObjects.nearestObject(1., 1., 1.));

    printf("  construction and accessors: ok\n");
  }


  // get_cell, get_cellCoords and get_objectsInCell over the whole grid, and
  // the box's closed upper end.
  void testCells ()
  {
    const std::vector<double> X = {0.5}, Y = {0.5}, Z = {0.5};
    const MeshGrid grid(X, Y, Z, 1., boxOf({{0., 10.}, {0., 4.}, {0., 2.}}));

    assert(grid.get_nCells()[0] == 10);
    assert(grid.get_nCells()[1] == 4);
    assert(grid.get_nCells()[2] == 2);

    const auto first = grid.get_cellCoords(0., 0., 0.);
    assert(first[0] == 0 && first[1] == 0 && first[2] == 0);
    assert(grid.get_cell(0., 0., 0.) == 0);

    // The upper face. The box is closed at both ends, so (10,4,2), the far
    // corner, exactly on the upper face of all three axes, is inside it and
    // must be accepted. Dividing by the cellsize there gives 10, 4 and 2,
    // which are one past the last cell on each axis, so the point belongs to
    // the last cell, (9,3,1), and not to a cell that does not exist.
    const auto last = grid.get_cellCoords(10., 4., 2.);
    assert(last[0] == 9 && last[1] == 3 && last[2] == 1);
    assert(grid.get_cell(10., 4., 2.) == 9*4*2+3*2+1);

    // One axis at a time, in case an axis is handled separately.
    const auto edgeX = grid.get_cellCoords(10., 0.5, 0.5);
    assert(edgeX[0] == 9);
    const auto edgeY = grid.get_cellCoords(0.5, 4., 0.5);
    assert(edgeY[1] == 3);
    const auto edgeZ = grid.get_cellCoords(0.5, 0.5, 2.);
    assert(edgeZ[2] == 1);

    const auto mid = grid.get_cellCoords(3.7, 2.2, 1.5);
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
          const auto coords = grid.get_cellCoords(x, y, z);
          assert(coords[0] == ix && coords[1] == iy && coords[2] == iz);
          const size_t linear = grid.get_cell(x, y, z);
          assert(linear < total);
          assert(seen[linear] == 0);
          seen[linear] = 1;
        }

    const auto inCell = grid.get_objectsInCell(0.9, 0.9, 0.9);
    assert(inCell.size() == 1 && inCell[0] == 0);
    assert(grid.get_objectsInCell(5.5, 2.5, 1.5).empty());

    // Same thing through addObject and a query: an object added exactly on the
    // far corner of the box must land in the last cell and be findable there.
    MeshGrid onFace({0.}, {0.}, {0.}, 1., boxOf({{0., 3.}, {0., 3.}, {0., 3.}}));
    const unsigned int corner = onFace.addObject(3., 3., 3.);
    const auto cell = onFace.get_objectsInCell(3., 3., 3.);
    assert(cell.size() == 1 && cell[0] == corner);
    assert(onFace.get_cellCoords(3., 3., 3.)[0] == 2);
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

    ASSERT_THROWS(MeshGrid({1., 2.}, Y, Z, 1., boxOf(lims)));
    ASSERT_THROWS(MeshGrid(X, {1., 2.}, Z, 1., boxOf(lims)));
    ASSERT_THROWS(MeshGrid(X, Y, {1., 2.}, 1., boxOf(lims)));
    ASSERT_THROWS(MeshGrid(X, Y, Z, 0., boxOf(lims)));
    ASSERT_THROWS(MeshGrid(X, Y, Z, -1., boxOf(lims)));
    // That the limits are 3x2 is now a property of their type, so only their values are checked.
    ASSERT_THROWS(MeshGrid(X, Y, Z, 1., boxOf({{2., 4.}, {0., 4.}, {0., 4.}})));
    ASSERT_THROWS(MeshGrid(X, Y, Z, 1., boxOf({{0., 4.}, {0., 2.5}, {0., 4.}})));
    ASSERT_THROWS(MeshGrid(X, Y, Z, 1., boxOf({{0., 4.}, {0., 4.}, {1.5, 4.}})));
    ASSERT_OK(MeshGrid(X, Y, Z, 1., boxOf({{1., 3.}, {1., 3.}, {1., 3.}})));

    MeshGrid grid(X, Y, Z, 1., boxOf(lims));
    const MeshGrid& cgrid = grid;

    const double out = 5.;
    ASSERT_THROWS(cgrid.nearestObject(out, 1., 1.));
    ASSERT_THROWS(cgrid.nearestObject(1., out, 1.));
    ASSERT_THROWS(cgrid.nearestObject(1., 1., out));
    ASSERT_THROWS(cgrid.nearestObject(-1., 1., 1.));
    ASSERT_THROWS(cgrid.nearestObjects(1, out, 1., 1.));
    ASSERT_THROWS(cgrid.closeObjects(out, 1., 1., 1.));
    ASSERT_THROWS(cgrid.get_cell(out, 1., 1.));
    ASSERT_THROWS(cgrid.get_cellCoords(out, 1., 1.));
    ASSERT_THROWS(cgrid.get_objectsInCell(out, 1., 1.));
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
    ASSERT_THROWS(cgrid.get_cell(nan_coord, 1., 1.));
    ASSERT_THROWS(cgrid.get_cellCoords(1., nan_coord, 1.));
    ASSERT_THROWS(cgrid.get_objectsInCell(1., 1., nan_coord));
    ASSERT_THROWS(grid.addObject(nan_coord, 1., 1.));
    ASSERT_THROWS(cgrid.nearestObject(inf_radius, 1., 1.));
    ASSERT_THROWS(grid.addObject(1., -inf_radius, 1.));

    MeshGrid single({1.}, {1.}, {1.}, 1., boxOf(lims));
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

    MeshGrid removed({1., 2.}, {1., 2.}, {1., 2.}, 1., boxOf(lims));
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

    MeshGrid grid(cloud.X, cloud.Y, cloud.Z, 0.12, boxOf(lims));
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
      const auto cell = grid.get_objectsInCell(cloud.X[i], cloud.Y[i], cloud.Z[i]);
      assert(std::find(cell.begin(), cell.end(), (unsigned int)i) != cell.end());
    }

    for (const auto i : removed) {
      const auto cell = grid.get_objectsInCell(cloud.X[i], cloud.Y[i], cloud.Z[i]);
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

  // The constructor's input checks: every input the header names as invalid
  // raises Error, on each axis and at each end, whether the box is given or
  // fitted to the points. A NaN must be refused as well as an infinity, since
  // every comparison with it is false and a check written as a comparison
  // would let it through.
  void testConstructorValidation ()
  {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const std::vector<double> X = {1., 2., 3.};
    const std::vector<double> Y = {1., 2., 3.};
    const std::vector<double> Z = {1., 2., 3.};
    const std::vector<std::vector<double>> lims = {{0., 4.}, {0., 4.}, {0., 4.}};

    for (const double cellsize : {inf, -inf, nan, 0., -1.}) {
      ASSERT_THROWS_PLAIN(MeshGrid(X, Y, Z, cellsize, boxOf(lims)));
      ASSERT_THROWS_PLAIN(MeshGrid(X, Y, Z, cellsize));
    }

    for (int axis=0; axis<3; ++axis)
      for (int end=0; end<2; ++end)
        for (const double bad : {nan, inf, -inf}) {
          auto badLims = lims;
          badLims[axis][end] = bad;
          ASSERT_THROWS_PLAIN(MeshGrid(X, Y, Z, 1., boxOf(badLims)));
          ASSERT_THROWS_PLAIN(MeshGrid({}, {}, {}, 1., boxOf(badLims)));
        }

    for (int axis=0; axis<3; ++axis) {
      auto inverted = lims;
      inverted[axis] = {4., 0.};
      ASSERT_THROWS_PLAIN(MeshGrid(X, Y, Z, 1., boxOf(inverted)));
      ASSERT_THROWS_PLAIN(MeshGrid({}, {}, {}, 1., boxOf(inverted)));
    }

    for (int axis=0; axis<3; ++axis)
      for (const double bad : {nan, inf, -inf}) {
        std::vector<std::vector<double>> coords = {X, Y, Z};
        coords[axis][1] = bad;
        ASSERT_THROWS_PLAIN(MeshGrid(coords[0], coords[1], coords[2], 1., boxOf(lims)));
        ASSERT_THROWS_PLAIN(MeshGrid(coords[0], coords[1], coords[2], 1.));
      }

    ASSERT_OK(MeshGrid(X, Y, Z, 1., boxOf({{0., 4.}, {0., 4.}, {0., 4.}})));
    ASSERT_OK(MeshGrid({}, {}, {}, 1., boxOf({{2., 2.}, {0., 4.}, {0., 4.}})));

    // A box whose squared diagonal overflows a double is refused, since a
    // distance inside it could then overflow too. The first case is two
    // objects 9e159 apart, which used to make nearestObject report that no
    // object was found. A box of 6e153 per side, grown to at most 7e153,
    // stays representable and is accepted.
    ASSERT_THROWS_PLAIN(MeshGrid({0., 9.e159}, {0., 0.}, {0., 0.}, 1.e159));
    ASSERT_THROWS_PLAIN(MeshGrid({}, {}, {}, 1.e153, boxOf({{-1.e154, 1.e154}, {0., 0.}, {0., 0.}})));
    ASSERT_OK(MeshGrid({}, {}, {}, 1.e153, boxOf({{0., 6.e153}, {0., 6.e153}, {0., 6.e153}})));

    // A cellsize whose reciprocal overflows is refused. Only a box of zero
    // extent can have so few cells that such a cellsize passes the count
    // limits, so that is the case to check.
    ASSERT_THROWS_PLAIN(MeshGrid({0.}, {0.}, {0.}, 1.e-310));
    ASSERT_THROWS_PLAIN(MeshGrid({}, {}, {}, std::numeric_limits<double>::denorm_min(), boxOf({{1., 1.}, {1., 1.}, {1., 1.}})));
    ASSERT_OK(MeshGrid({0.}, {0.}, {0.}, 1.e-300));

    printf("  constructor input validation: ok\n");
  }


  // An axis of zero extent gets one cell and a box one cellsize wide centred
  // on it, as the header states, and the grid still answers every query as
  // brute force does. Built both ways the header names: points that are all
  // equal along the axis with no limits, and limits with lo == hi. A flat
  // cloud has one such axis, a needle two.
  void testZeroExtent ()
  {
    const double cellsize = 0.1;
    const double level = 0.37;
    std::mt19937 gen(31337u);
    std::uniform_real_distribution<double> unit(0., 1.);

    for (int flatAxes=1; flatAxes<=2; ++flatAxes) {
      Cloud cloud;
      for (int n=0; n<800; ++n) {
        cloud.X.push_back(unit(gen));
        cloud.Y.push_back(flatAxes == 2 ? level : unit(gen));
        cloud.Z.push_back(level);
      }
      cloud.alive.assign(cloud.size(), 1);

      const std::vector<std::vector<double>> lims =
        {{0., 1.}, flatAxes == 2 ? std::vector<double>{level, level} : std::vector<double>{0., 1.}, {level, level}};

      const MeshGrid fitted(cloud.X, cloud.Y, cloud.Z, cellsize);
      const MeshGrid given(cloud.X, cloud.Y, cloud.Z, cellsize, boxOf(lims));

      for (const MeshGrid* grid : {&fitted, &given}) {
        for (int axis=3-flatAxes; axis<3; ++axis) {
          assert(grid->get_nCells()[axis] == 1);
          const auto box = grid->get_lims();
          assert(std::fabs(box[axis][0]-(level-cellsize/2.)) < tol);
          assert(std::fabs(box[axis][1]-(level+cellsize/2.)) < tol);
        }

        const auto box = grid->get_lims();
        std::uniform_real_distribution<double> alongX(box[0][0], box[0][1]);
        std::uniform_real_distribution<double> alongY(box[1][0], box[1][1]);
        std::uniform_int_distribution<unsigned int> pick(0, (unsigned int)cloud.size()-1);
        for (int q=0; q<60; ++q) {
          const double x = alongX(gen);
          const double y = (flatAxes == 2) ? level : alongY(gen);
          checkNearest(*grid, cloud, x, y, level, -1);
          checkNearestN(*grid, cloud, 7, x, y, level, -1);
          checkShell(*grid, cloud, x, y, level, 0.08, 0., -1);
          checkShell(*grid, cloud, x, y, level, 0.2, 0.1, -1);

          const unsigned int index = pick(gen);
          checkNearest(*grid, cloud, cloud.X[index], cloud.Y[index], cloud.Z[index], (long)index);
          checkNearestN(*grid, cloud, 7, cloud.X[index], cloud.Y[index], cloud.Z[index], (long)index);
          checkShell(*grid, cloud, cloud.X[index], cloud.Y[index], cloud.Z[index], 0.08, 0., (long)index);
        }
      }
    }

    printf("  zero-extent axes, flat and needle: ok\n");
  }


  // A grid too large to count is refused with Error, and refused before
  // anything is allocated: each of these would need from terabytes to far more
  // memory than exists, so reaching the end of this group at all, quickly and
  // without std::bad_alloc, is what shows the checks come first. The cases are
  // a cellsize tiny against the box, one axis a single cell past 2147483647,
  // and 2^22 cells on every axis, whose 2^66 cells overflow a 64-bit count
  // although each axis is well within its own limit.
  void testCountLimits ()
  {
    const std::vector<std::vector<double>> unitBox = {{0., 1.}, {0., 1.}, {0., 1.}};
    ASSERT_THROWS_PLAIN(MeshGrid({0.5}, {0.5}, {0.5}, 1.e-300, boxOf(unitBox)));
    ASSERT_THROWS_PLAIN(MeshGrid({}, {}, {}, 1.e-300, boxOf(unitBox)));

    const double pastIntMax = 2147483648.;
    ASSERT_THROWS_PLAIN(MeshGrid({}, {}, {}, 1., boxOf({{0., pastIntMax}, {0., 1.}, {0., 1.}})));
    ASSERT_THROWS_PLAIN(MeshGrid({}, {}, {}, 1., boxOf({{0., 1.}, {0., 1.}, {0., pastIntMax}})));

    const double side = 4194304.;
    ASSERT_THROWS_PLAIN(MeshGrid({}, {}, {}, 1., boxOf({{0., side}, {0., side}, {0., side}})));

    const double huge = 1.e308;
    ASSERT_THROWS_PLAIN(MeshGrid({-huge, huge}, {0., 1.}, {0., 1.}, 1.));

    printf("  grids too large to count: ok\n");
  }


  // A needle of 46341 x 1 x 1 cells. 46341^2 exceeds the largest 32-bit
  // signed integer, which is where 0.1.0's cell-offset arithmetic overflowed;
  // the grid itself is small (a few MB), so this is the cheapest grid on the
  // far side of that threshold. Two clouds: a few thousand points along the
  // needle, and two points at its two ends, which forces the nearest-object
  // searches to walk the whole length and so to use the offsets whose squared
  // length is past the threshold.
  void testLongAxis ()
  {
    const double length = 46341.;
    const std::vector<std::vector<double>> lims = {{0., length}, {0., 1.}, {0., 1.}};
    std::mt19937 gen(46341u);
    std::uniform_real_distribution<double> along(0., length);
    std::uniform_real_distribution<double> unit(0., 1.);

    Cloud cloud;
    for (int n=0; n<3000; ++n) {
      cloud.X.push_back(along(gen));
      cloud.Y.push_back(unit(gen));
      cloud.Z.push_back(unit(gen));
    }
    cloud.alive.assign(cloud.size(), 1);

    const MeshGrid grid(cloud.X, cloud.Y, cloud.Z, 1., boxOf(lims));
    assert(grid.get_nCells()[0] == 46341);

    for (int q=0; q<40; ++q) {
      const double x = along(gen), y = unit(gen), z = unit(gen);
      checkNearest(grid, cloud, x, y, z, -1);
      checkNearestN(grid, cloud, 9, x, y, z, -1);
      checkShell(grid, cloud, x, y, z, 40., 0., -1);
      checkShell(grid, cloud, x, y, z, 30000., 20000., -1);
    }
    checkShell(grid, cloud, 0., 0.5, 0.5, std::numeric_limits<double>::infinity(), 46000., -1);

    Cloud ends;
    ends.X = {0.25, length-0.25};
    ends.Y = {0.5, 0.5};
    ends.Z = {0.5, 0.5};
    ends.alive.assign(2, 1);

    const MeshGrid sparse(ends.X, ends.Y, ends.Z, 1., boxOf(lims));
    assert(sparse.nearestObject(0u) == 1);
    assert(sparse.nearestObject(1u) == 0);
    checkNearest(sparse, ends, 1., 0.5, 0.5, -1);
    checkNearestN(sparse, ends, 2, 1., 0.5, 0.5, -1);
    checkNearestN(sparse, ends, 1, ends.X[0], ends.Y[0], ends.Z[0], 0);
    checkShell(sparse, ends, 0., 0.5, 0.5, length+1., length-1., -1);

    printf("  a 46341-cell axis, past the 32-bit offset threshold: ok\n");
  }


  // Not tested here, because no test can reach them at a reasonable cost:
  // the 4294967295-object limit of the constructor and of addObject, which
  // needs that many objects, and addObject leaving the grid unchanged when an
  // allocation fails, which needs an allocator the grid does not let a test
  // replace. Both are covered by review.


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
      const MeshGrid grid(cloud.X, cloud.Y, cloud.Z, cellsize, boxOf(lims));

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
      const MeshGrid grid({10.75, 13.}, {10.5, 10.5}, {10.5, 10.5}, 1., boxOf(lims));
      assert(grid.get_cellCoords(10.75, 10.5, 10.5)[0] == 10);
      assert(grid.get_cellCoords(13., 10.5, 10.5)[0] == 13);
      assert(sorted(grid.closeObjects(10.75, 10.5, 10.5, 2.25)) == std::vector<unsigned int>({0, 1}));
      assert(sorted(grid.closeObjects(0u, 2.25)) == std::vector<unsigned int>({1}));
      assert(grid.closeObjects(0u, 2.25, 2.25).size() == 1);
      // a hair inside the edge and the object is correctly out of the shell
      assert(grid.closeObjects(0u, 2.2421875).empty());
    }

    // The same at layer 4: offset 5, gap 4, distance 4.25 from x = 10.75 to
    // x = 15, with Rmax = 4.25 so floor(Rmax/cellsize) = 4.
    {
      const MeshGrid grid({10.75, 15.}, {10.5, 10.5}, {10.5, 10.5}, 1., boxOf(lims));
      assert(sorted(grid.closeObjects(0u, 4.25)) == std::vector<unsigned int>({1}));
      assert(grid.closeObjects(0u, 4.25, 4.25).size() == 1);
    }

    // Off axis, where all three gaps contribute: offset (3,3,3), gaps (2,2,2),
    // layer floor(sqrt(12)) = 3, distance 2.25*sqrt(3) = 3.897. Rmax = 3.9375
    // keeps floor(Rmax/cellsize) = 3, so layer 3 is again the last scanned.
    {
      const MeshGrid grid({10.75, 13.}, {10.75, 13.}, {10.75, 13.}, 1., boxOf(lims));
      const double d = std::sqrt(3.*2.25*2.25);
      assert(d > 3. && d < 3.9375);
      assert(sorted(grid.closeObjects(0u, 3.9375)) == std::vector<unsigned int>({1}));
    }

    printf("  tightest layer the Rmax bound admits: ok\n");
  }


  // The load-bearing group: every query kind against brute force over several
  // thousand random points, at three cell sizes relative to the mean
  // separation. This is what establishes that the geometry is right, rather
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

      const MeshGrid grid(cloud.X, cloud.Y, cloud.Z, cellsize, boxOf(lims));
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
    const MeshGrid grid(cloud.X, cloud.Y, cloud.Z, cellsize, boxOf(lims));

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
                               boxOf({{0., 40.}, {0., 40.}, {0., 40.}}));

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
  // hand-placed points whose distances are exact in binary so neither boundary
  // comparison is decided by rounding: an object at exactly Rmin is returned,
  // and so is one at exactly Rmax.
  //
  // The interval is not where self-exclusion lives. The index overload drops
  // the query object by identity and nothing else, so a coincident duplicate
  // at d = 0 is a different object and is returned; the point overload has no
  // identity to drop, so it returns everything at d = 0.
  void testRminBoundary ()
  {
    const std::vector<double> X = {5., 6., 7., 8., 5., 5.};
    const std::vector<double> Y = {5., 5., 5., 5., 7., 8.};
    const std::vector<double> Z = {5., 5., 5., 5., 5., 5.};
    const std::vector<std::vector<double>> lims = {{0., 16.}, {0., 16.}, {0., 16.}};

    const MeshGrid grid(X, Y, Z, 1., boxOf(lims));

    auto sorted = [](std::vector<unsigned int> v) {
      std::sort(v.begin(), v.end());
      return v;
    };

    assert(sorted(grid.closeObjects(5., 5., 5., 2., 2.)) == std::vector<unsigned int>({2, 4}));
    assert(sorted(grid.closeObjects(0u, 2., 2.)) == std::vector<unsigned int>({2, 4}));
    assert(sorted(grid.closeObjects(0u, 3., 2.)) == std::vector<unsigned int>({2, 3, 4, 5}));
    assert(sorted(grid.closeObjects(0u, 3., 3.)) == std::vector<unsigned int>({3, 5}));

    // a ball takes in everything out to Rmax, the objects at exactly Rmax
    // among them, and from a point it takes in the object at d = 0 too
    assert(sorted(grid.closeObjects(0u, 2.)) == std::vector<unsigned int>({1, 2, 4}));
    assert(sorted(grid.closeObjects(5., 5., 5., 2.)) == std::vector<unsigned int>({0, 1, 2, 4}));
    assert(sorted(grid.closeObjects(5., 5., 5., 3., 3.)) == std::vector<unsigned int>({3, 5}));

    assert(sorted(grid.closeObjects(0u, 0., 0.)).empty());
    assert(sorted(grid.closeObjects(5., 5., 5., 0., 0.)) == std::vector<unsigned int>({0}));

    // three objects share a position: the query object is excluded by
    // identity, the two duplicates at d = 0 are not
    const std::vector<double> dX = {2., 2., 2., 4.};
    const std::vector<double> dY = {2., 2., 2., 2.};
    const std::vector<double> dZ = {2., 2., 2., 2.};
    const MeshGrid dup(dX, dY, dZ, 1., boxOf(lims));

    assert(sorted(dup.closeObjects(2., 2., 2., 0.)) == std::vector<unsigned int>({0, 1, 2}));
    assert(sorted(dup.closeObjects(0u, 0.)) == std::vector<unsigned int>({1, 2}));
    assert(sorted(dup.closeObjects(0u, 0., 0.)) == std::vector<unsigned int>({1, 2}));
    assert(dup.nearestObject(0u) == 1 || dup.nearestObject(0u) == 2);
    assert(dup.nearestObjects(2, 0u).size() == 2);
    assert(dup.nearestObjects(2, 0u) == std::vector<unsigned int>({1, 2}));
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
      const MeshGrid grid(cloud.X, cloud.Y, cloud.Z, cellsize, boxOf(lims));
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


  // "Nearest first; ties by ascending index", on inputs where most distances
  // tie exactly: integer lattices, where every squared distance is a small
  // exact integer.
  void testTieOrder ()
  {
    // A 3x3x3 lattice at integer coordinates, object a*9+b*3+c at (a, b, c),
    // worked by hand. From the centre (1, 1, 1), object 13: six objects at
    // d^2 = 1 (4, 10, 12, 14, 16, 22), twelve at d^2 = 2 (1, 3, 5, 7, 9, 11,
    // 15, 17, 19, 21, 23, 25). Asking for 6 + 2 cuts the d^2 = 2 shell, so the
    // last two places must go to its two lowest indices, 1 and 3. From
    // (0.5, 0.5, 0.5) the eight corners of the unit cube all tie, and the
    // three lowest of them are kept. Cellsize 1.5 puts several objects in a
    // cell, so the grid's internal order is not the index order.
    std::vector<double> X, Y, Z;
    for (int a=0; a<3; ++a) for (int b=0; b<3; ++b) for (int c=0; c<3; ++c) {
      X.push_back(a); Y.push_back(b); Z.push_back(c);
    }
    const std::vector<std::vector<double>> small = {{0., 3.}, {0., 3.}, {0., 3.}};
    const MeshGrid grid(X, Y, Z, 1.5, boxOf(small));

    assert(grid.nearestObjects(8, 13u) == std::vector<unsigned int>({4, 10, 12, 14, 16, 22, 1, 3}));
    assert(grid.nearestObjects(9, 1., 1., 1.) == std::vector<unsigned int>({13, 4, 10, 12, 14, 16, 22, 1, 3}));
    assert(grid.nearestObjects(3, 0.5, 0.5, 0.5) == std::vector<unsigned int>({0, 1, 3}));
    assert(grid.nearestObjects(8, 0.5, 0.5, 0.5) == std::vector<unsigned int>({0, 1, 3, 4, 9, 10, 12, 13}));

    // A 12^3 lattice whose indices are a random permutation of the sites, so
    // that index order has nothing to do with position or with the grid's
    // cell order; then a few removals, and additions coincident with existing
    // sites, whose new indices are higher than every original one and must
    // therefore come after the objects they tie with. N runs through values
    // that end inside a shell (a tie at the N-th place) and the 113 otswap
    // uses; cellsizes from below the spacing to several spacings.
    const int side = 12;
    const size_t n = size_t(side)*side*side;
    std::mt19937 gen(8u);
    std::vector<size_t> site(n);
    for (size_t i=0; i<n; ++i) site[i] = i;
    std::shuffle(site.begin(), site.end(), gen);

    const std::vector<std::vector<double>> box = {{0., double(side)}, {0., double(side)}, {0., double(side)}};
    const double cellsizes[] = {0.7, 1., 2.5, 4.};
    const unsigned int counts[] = {1, 2, 5, 7, 13, 20, 27, 60, 113};

    for (const double cellsize : cellsizes) {
      Cloud cloud;
      for (size_t i=0; i<n; ++i) {
        cloud.X.push_back(double(site[i]/(side*side)));
        cloud.Y.push_back(double(site[i]/side%side));
        cloud.Z.push_back(double(site[i]%side));
      }
      cloud.alive.assign(n, 1);
      MeshGrid lattice(cloud.X, cloud.Y, cloud.Z, cellsize, boxOf(box));

      std::uniform_int_distribution<unsigned int> pick(0, (unsigned int)n-1);
      for (int r=0; r<40; ++r) {
        const unsigned int index = pick(gen);
        if (!cloud.alive[index]) continue;
        lattice.removeObject(index);
        cloud.alive[index] = 0;
      }
      for (int r=0; r<40; ++r) {
        const unsigned int source = pick(gen);
        const unsigned int added = lattice.addObject(cloud.X[source], cloud.Y[source], cloud.Z[source]);
        assert(added == cloud.size());
        cloud.X.push_back(cloud.X[source]);
        cloud.Y.push_back(cloud.Y[source]);
        cloud.Z.push_back(cloud.Z[source]);
        cloud.alive.push_back(1);
      }

      std::uniform_int_distribution<int> coord(0, 2*side);
      std::uniform_int_distribution<unsigned int> anyIndex(0, (unsigned int)cloud.size()-1);
      for (int q=0; q<30; ++q) {
        // half-integer coordinates: lattice sites, cell centres, edge and
        // face midpoints, each with its own pattern of ties
        const double x = 0.5*coord(gen), y = 0.5*coord(gen), z = 0.5*coord(gen);
        unsigned int index = anyIndex(gen);
        while (!cloud.alive[index]) index = anyIndex(gen);
        for (const unsigned int N : counts) {
          checkNearestN(lattice, cloud, N, x, y, z, -1);
          checkNearestN(lattice, cloud, N, cloud.X[index], cloud.Y[index], cloud.Z[index], (long)index);
        }
      }
      printf("  tie order on a lattice (cellsize %.1f): ok\n", cellsize);
    }
  }


  // The stop and range rules near cell faces. A point within a few ulp of a
  // face can be assigned to the cell on the far side of it, so a cell's
  // contents may reach slightly closer to the query than the cell's nominal
  // faces suggest. The header promises exact membership and the true nearest
  // object regardless, so each case below is a configuration where a rule
  // with no allowance for this loses a result. Each was found by search and
  // is checked here exactly as found.
  void testCellFaceRounding ()
  {
    const double inf = std::numeric_limits<double>::infinity();

    // Cellsize 0.1 on [0, 0.8]. 0.7*10 rounds to exactly 7, so the object at
    // 0.7 sits in cell 7 although the double 0.7 lies just below 7*0.1. The
    // query just below 0.4 is in cell 3: offset 4, gap 3, shell 3. The
    // difference 0.7 - q is the double 0.3 exactly, so the object is at
    // exactly Rmax = 0.3, where floor(Rmax/cellsize) = 2 would stop the walk
    // one shell short.
    {
      const std::vector<std::vector<double>> lims = {{0., 0.8}, {0., 0.}, {0., 0.}};
      const double q = std::nextafter(0.4, 0.);
      assert(0.7-q == 0.3);

      const MeshGrid grid({0.7}, {0.}, {0.}, 0.1, boxOf(lims));
      assert(grid.get_cellCoords(0.7, 0., 0.)[0] == 7);
      assert(grid.get_cellCoords(q, 0., 0.)[0] == 3);
      assert(grid.closeObjects(q, 0., 0., 0.3) == std::vector<unsigned int>({0}));
      assert(grid.closeObjects(q, 0., 0., 0.3, 0.3) == std::vector<unsigned int>({0}));

      const MeshGrid pair({q, 0.7}, {0., 0.}, {0., 0.}, 0.1, boxOf(lims));
      assert(pair.closeObjects(0u, 0.3) == std::vector<unsigned int>({1}));
      assert(pair.closeObjects(1u, 0.3) == std::vector<unsigned int>({0}));

      // Object A, at 0x1.9999999999996p-4 just below 0.1 in cell 0, is in
      // shell 2 from the same query; B, at 0.7, is in shell 3 and is nearer
      // both exactly and in the computed squared distance. A stop with no
      // margin, (3*cellsize)^2 >= the best so far, ends the walk before
      // shell 3 and returns A.
      const double a = 0x1.9999999999996p-4;
      const double qa = q-a, bq = 0.7-q;
      assert(bq*bq < qa*qa);

      const MeshGrid two({a, 0.7}, {0., 0.}, {0., 0.}, 0.1, boxOf(lims));
      assert(two.nearestObject(q, 0., 0.) == 1);
      assert(two.nearestObjects(1, q, 0., 0.) == std::vector<unsigned int>({1}));

      const MeshGrid three({a, 0.7, q}, {0., 0., 0.}, {0., 0., 0.}, 0.1, boxOf(lims));
      assert(three.nearestObject(2u) == 1);
      assert(three.nearestObjects(1, 2u) == std::vector<unsigned int>({1}));
    }

    // The lower end of a shell, with a cellsize of about one ulp of the
    // coordinates, which are near -9.8e7. The top cell of such a grid can
    // overhang its nominal face by most of a cell, so an object can lie
    // further from the query than the shell index suggests. Here a single
    // object lies at exactly Rmin from a query on the box's upper corner,
    // and a range starting at floor(Rmin/cellsize) - 4 never reaches it.
    {
      const double c = 0x1.457ef4d38d4ebp-26;
      const std::vector<std::vector<double>> lims =
        {{-0x1.74a7f8cac18dp+26, -0x1.74a7f8cac18bep+26},
         {-0x1.74a7f8cac1876p+26, -0x1.74a7f8cac1868p+26},
         {-0x1.74a7f8cac18cp+26, -0x1.74a7f8cac18aep+26}};
      const double ox = -0x1.74a7f8cac18cdp+26, oy = -0x1.74a7f8cac1874p+26, oz = -0x1.74a7f8cac18bep+26;
      const double qx = -0x1.74a7f8cac18bdp+26, qy = -0x1.74a7f8cac1867p+26, qz = -0x1.74a7f8cac18adp+26;
      const double Rmin = 0x1.ab884ef702284p-22;
      const double dx = ox-qx, dy = oy-qy, dz = oz-qz;
      assert(dx*dx+dy*dy+dz*dz >= Rmin*Rmin);

      const MeshGrid grid({ox}, {oy}, {oz}, c, boxOf(lims));
      assert(grid.closeObjects(qx, qy, qz, inf, Rmin) == std::vector<unsigned int>({0}));

      MeshGrid added = grid;
      const unsigned int query = added.addObject(qx, qy, qz);
      assert(added.closeObjects(query, inf, Rmin) == std::vector<unsigned int>({0}));
    }

    printf("  rounding at cell faces, found by search: ok\n");
  }


  // Every query kind against brute force on points placed within a few ulp of
  // cell faces, where rounding decides which cell a point falls in, with radii
  // chosen so that an object lies exactly on Rmax or Rmin. Two regimes: an
  // ordinary cellsize, and one of about an ulp of the coordinates, where the
  // top cell can overhang its nominal face by most of a cell. Objects are
  // also added and removed, so the cells that addObject and removeObject
  // compute are exercised against the same rounding.
  void testFaceHugging ()
  {
    std::mt19937_64 gen(2611u);
    auto uniform = [&](const double a, const double b) { return std::uniform_real_distribution<double>(a, b)(gen); };
    auto integer = [&](const int a, const int b) { return std::uniform_int_distribution<int>(a, b)(gen); };
    auto ulp = [](const double x) { return std::nextafter(std::fabs(x), std::numeric_limits<double>::infinity())-std::fabs(x); };
    auto nudge = [](double x, const int steps) {
      for (int k=0; k<std::abs(steps); ++k)
        x = std::nextafter(x, steps > 0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity());
      return x;
    };

    // The smallest radius whose square, as computed, is >= d2, and the largest
    // whose square is <= d2: an object at computed squared distance d2 is then
    // exactly on Rmax, or exactly on Rmin.
    auto radiusAbove = [](const double d2) {
      double R = std::sqrt(d2);
      while (R*R < d2) R = std::nextafter(R, std::numeric_limits<double>::infinity());
      for (double P = std::nextafter(R, 0.); P != R && P*P >= d2; P = std::nextafter(R, 0.)) R = P;
      return R;
    };
    auto radiusBelow = [](const double d2) {
      if (d2 == 0.) return 0.;
      double R = std::sqrt(d2);
      while (R*R > d2) R = std::nextafter(R, 0.);
      for (double P = std::nextafter(R, std::numeric_limits<double>::infinity()); P*P <= d2;
           P = std::nextafter(R, std::numeric_limits<double>::infinity())) R = P;
      return R;
    };

    int grids = 0;
    for (int trial=0; trial<600; ++trial) {
      const bool ulpScale = (trial%2 == 1);
      double c;
      std::vector<std::vector<double>> lims(3);
      if (!ulpScale) {
        c = uniform(0.01, 10.);
        for (int a=0; a<3; ++a) {
          const double lo = uniform(-100., 100.);
          lims[a] = {lo, lo+c*integer(1, 8)+(integer(0, 1) ? 0. : uniform(0., c))};
        }
      }
      else {
        const double base = std::ldexp(uniform(1., 2.), integer(20, 40))*(integer(0, 1) ? 1. : -1.);
        const double u = ulp(base);
        c = u*uniform(0.6, 1.6);
        for (int a=0; a<3; ++a) {
          const double lo = base+u*integer(-50, 50);
          lims[a] = {lo, lo+u*integer(8, 20)};
        }
      }

      const MeshGrid probe({}, {}, {}, c, boxOf(lims));
      const auto box = probe.get_lims();
      auto pick = [&](const int a) {
        double x;
        if (!ulpScale) x = nudge(box[a][0]+integer(0, int(probe.get_nCells()[a]))*c, integer(-3, 3));
        else x = lims[a][0]+ulp(lims[a][0])*integer(0, 4);
        if (integer(0, 5) == 0) x = integer(0, 1) ? lims[a][0] : lims[a][1];
        return std::min(std::max(x, lims[a][0]), lims[a][1]);
      };

      Cloud cloud;
      const int n = integer(2, 30);
      for (int i=0; i<n; ++i) {
        cloud.X.push_back(pick(0));
        cloud.Y.push_back(pick(1));
        cloud.Z.push_back(pick(2));
      }
      cloud.alive.assign(n, 1);
      MeshGrid grid(cloud.X, cloud.Y, cloud.Z, c, boxOf(lims));
      ++grids;

      for (int r=0; r<3; ++r) {
        const unsigned int victim = (unsigned int)integer(0, n-1);
        if (!cloud.alive[victim] || grid.get_nObjects() < 3) continue;
        grid.removeObject(victim);
        cloud.alive[victim] = 0;
      }
      for (int r=0; r<3; ++r) {
        const double x = pick(0), y = pick(1), z = pick(2);
        assert(grid.addObject(x, y, z) == cloud.size());
        cloud.X.push_back(x); cloud.Y.push_back(y); cloud.Z.push_back(z); cloud.alive.push_back(1);
      }

      for (size_t i=0; i<cloud.size(); ++i) {
        if (!cloud.alive[i]) continue;
        const auto own = grid.get_objectsInCell(cloud.X[i], cloud.Y[i], cloud.Z[i]);
        assert(std::find(own.begin(), own.end(), (unsigned int)i) != own.end());
      }

      for (int q=0; q<6; ++q) {
        long skip = -1;
        double x = pick(0), y = pick(1), z = pick(2);
        if (q%3 == 1) { x = box[0][1]; y = box[1][1]; z = box[2][1]; }
        if (q%3 == 2) {
          skip = integer(0, int(cloud.size())-1);
          if (!cloud.alive[skip]) continue;
          x = cloud.X[skip]; y = cloud.Y[skip]; z = cloud.Z[skip];
        }
        const auto ranked = cloud.ranked(x, y, z, skip);
        if (ranked.empty()) continue;

        const unsigned int nearest = (skip < 0) ? grid.nearestObject(x, y, z) : grid.nearestObject((unsigned int)skip);
        assert(cloud.dist_sq(nearest, x, y, z) == ranked.front().first);
        checkNearestN(grid, cloud, (unsigned int)integer(1, int(ranked.size())), x, y, z, skip);

        const double d2 = ranked[size_t(integer(0, int(ranked.size())-1))].first;
        checkShell(grid, cloud, x, y, z, radiusAbove(d2), 0., skip);
        checkShell(grid, cloud, x, y, z, std::numeric_limits<double>::infinity(), radiusBelow(d2), skip);
      }
    }
    assert(grids == 600);

    printf("  points on cell faces, every query against brute force: ok\n");
  }


  // The grown box contains the limits it was given, so a point on a limit is
  // accepted, as the header promises. Growing each side by half of
  // (cells*cellsize - range) can come out negative when range/cellsize rounds
  // down onto an integer; the first case is one where it did, and the sweep
  // runs ranges within a few ulp of a whole number of cells, where that
  // happens about once in two thousand.
  void testBoxContainsLimits ()
  {
    {
      const double c = 5.078493365404232, hi = 3279294.892895552;
      const std::vector<std::vector<double>> lims = {{0., hi}, {0., 1.}, {0., 1.}};
      const MeshGrid grid({0., hi}, {0.5, 0.5}, {0.5, 0.5}, c, boxOf(lims));
      const auto box = grid.get_lims();
      assert(box[0][0] <= 0. && box[0][1] >= hi);
      assert(grid.nearestObject(0., 0.5, 0.5) == 0);
      assert(grid.nearestObject(hi, 0.5, 0.5) == 1);
    }

    std::mt19937 gen(4242u);
    std::uniform_real_distribution<double> cellsizes(0.01, 10.), lows(-1.e4, 1.e4);
    std::uniform_int_distribution<int> counts(1, 200), steps(-3, 3);
    for (int t=0; t<20000; ++t) {
      const double c = cellsizes(gen), lo = lows(gen);
      double hi = lo+counts(gen)*c;
      const int k = steps(gen);
      for (int s=0; s<std::abs(k); ++s) hi = std::nextafter(hi, k > 0 ? 1.e300 : -1.e300);
      if (hi < lo) continue;

      const std::vector<std::vector<double>> lims = {{lo, hi}, {lo, hi}, {0., 0.}};
      const MeshGrid grid({lo, hi}, {hi, lo}, {0., 0.}, c, boxOf(lims));
      const auto box = grid.get_lims();
      assert(box[0][0] <= lo && box[0][1] >= hi);
      assert(box[1][0] <= lo && box[1][1] >= hi);
    }

    printf("  the grown box contains the limits: ok\n");
  }


  // The extremes of scale the constructor accepts, against brute force with
  // exact comparisons. Tiny: coordinates near 1e-160 and a cellsize near
  // 1e-161, where squared distances are subnormal. Huge: a box near 1e153 on
  // every side, whose squared diagonal is close to the largest double. Every
  // distance in an accepted box is finite, which is what makes the brute-force
  // reference meaningful there.
  void testExtremeScales ()
  {
    const double scales[] = {1.e-160, 1.e153};
    std::mt19937 gen(160u);
    std::uniform_real_distribution<double> unit(0., 1.);

    for (const double scale : scales) {
      const std::vector<std::vector<double>> lims = {{-scale, scale}, {-scale, scale}, {-scale, scale}};
      Cloud cloud;
      for (int i=0; i<300; ++i) {
        cloud.X.push_back(scale*(2.*unit(gen)-1.));
        cloud.Y.push_back(scale*(2.*unit(gen)-1.));
        cloud.Z.push_back(scale*(2.*unit(gen)-1.));
      }
      cloud.X.push_back(scale); cloud.Y.push_back(scale); cloud.Z.push_back(scale);
      cloud.X.push_back(-scale); cloud.Y.push_back(-scale); cloud.Z.push_back(-scale);
      cloud.alive.assign(cloud.size(), 1);

      const MeshGrid grid(cloud.X, cloud.Y, cloud.Z, scale/5., boxOf(lims));
      std::uniform_int_distribution<unsigned int> pick(0, (unsigned int)cloud.size()-1);

      for (int q=0; q<40; ++q) {
        const double x = scale*(2.*unit(gen)-1.), y = scale*(2.*unit(gen)-1.), z = scale*(2.*unit(gen)-1.);
        const auto ranked = cloud.ranked(x, y, z, -1);
        assert(cloud.dist_sq(grid.nearestObject(x, y, z), x, y, z) == ranked.front().first);
        checkNearestN(grid, cloud, 7, x, y, z, -1);
        checkShell(grid, cloud, x, y, z, 0.3*scale, 0., -1);
        checkShell(grid, cloud, x, y, z, 2.*scale, 1.5*scale, -1);

        const unsigned int index = pick(gen);
        const double px = cloud.X[index], py = cloud.Y[index], pz = cloud.Z[index];
        const auto own = cloud.ranked(px, py, pz, (long)index);
        assert(cloud.dist_sq(grid.nearestObject(index), px, py, pz) == own.front().first);
        checkNearestN(grid, cloud, 7, px, py, pz, (long)index);
        checkShell(grid, cloud, px, py, pz, 0.3*scale, 0., (long)index);
      }

      const unsigned int far = (unsigned int)cloud.size()-1;
      assert(grid.nearestObjects((unsigned int)cloud.size()-1, far).size() == cloud.size()-1);
      assert(grid.closeObjects(far, std::numeric_limits<double>::infinity()).size() == cloud.size()-1);
    }

    printf("  extreme scales against brute force: ok\n");
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
  testTieOrder();
  testCellFaceRounding();
  testFaceHugging();
  testBoxContainsLimits();
  testExtremeScales();
  testDegenerateExtents();
  testConstructorValidation();
  testZeroExtent();
  testCountLimits();
  testLongAxis();
  testWideShells();
  testShellRadiusSweep();
  testTightestScannedLayer();
  testBruteForce();

  printf("all tests passed\n");

  return 0;
}
