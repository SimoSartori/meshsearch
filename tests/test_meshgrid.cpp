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

    const auto last = grid.get_CellCoords(10., 4., 2.);
    assert(last[0] == 9 && last[1] == 3 && last[2] == 1);
    assert(grid.get_Cell(10., 4., 2.) == 9*4*2+3*2+1);

    const auto edgeX = grid.get_CellCoords(10., 0.5, 0.5);
    assert(edgeX[0] == 9);
    const auto edgeY = grid.get_CellCoords(0.5, 4., 0.5);
    assert(edgeY[1] == 3);
    const auto edgeZ = grid.get_CellCoords(0.5, 0.5, 2.);
    assert(edgeZ[2] == 1);

    const auto mid = grid.get_CellCoords(3.7, 2.2, 1.5);
    assert(mid[0] == 3 && mid[1] == 2 && mid[2] == 1);

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

    MeshGrid onFace({0.}, {0.}, {0.}, 1., {{0., 3.}, {0., 3.}, {0., 3.}});
    const unsigned int corner = onFace.addObject(3., 3., 3.);
    const auto cell = onFace.get_ObjectsInCell(3., 3., 3.);
    assert(cell.size() == 1 && cell[0] == corner);
    assert(onFace.get_CellCoords(3., 3., 3.)[0] == 2);
    assert(onFace.nearestObject(3., 3., 3.) == corner);
    assert(onFace.nearestObject(2.9, 2.9, 2.9) == corner);

    printf("  cell lookup and box faces: ok\n");
  }


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

    for (int k=2; k<=20; ++k) {
      const double Rmin = double(k);
      for (int step=0; step<24; ++step) {
        const double delta = 0.5*step;
        checkShell(corner_grid, corner, 10., 10., 10., Rmin+delta, Rmin, -1);
      }
    }

    printf("  wide shells at large Rmin: ok\n");
  }


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
  testBruteForce();

  printf("all tests passed\n");

  return 0;
}
