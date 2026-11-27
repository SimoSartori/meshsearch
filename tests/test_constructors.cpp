// Tests for the pointer constructors of meshsearch::MeshGrid: three arrays
// MeshGrid(X, Y, Z, n, cellsize[, opt]) and one interleaved array
// MeshGrid(XYZ, n, cellsize[, opt]). The header promises that each builds the
// grid the vector constructor builds from the same values; these tests check
// that promise, and the errors.
//
//   - Same grid: the same box, cells and cell side, the same objects in every
//     cell in the same order, and the same answer, in the same order, to
//     every query family, single and batched, point and index. Checked on
//     random data in the dense and the sparse regime, clustered data, data
//     with coincident points and signed zeros (which decide the fitted box
//     bit for bit), and on fitted, explicit, periodic and mixed boxes. The
//     vector grid is itself checked against brute force in test_meshgrid and
//     test_periodic, so equality with it is what these forms must show.
//   - The grid keeps no reference to the caller's arrays: they are
//     overwritten after the build, and the answers do not change.
//   - addObject and removeObject behave the same on all three grids.
//   - Errors: the same Error, with the same message, as the vector form for
//     every invalid input it shares; a null pointer with n > 0 raises Error,
//     and with n == 0 it is accepted.
//
// NDEBUG is undefined below so that assert() stays live whatever the build
// type.

#undef NDEBUG

#include "meshsearch/MeshGrid.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

using meshsearch::MeshGrid;
using Lims = std::array<std::array<double,2>,3>;

namespace {

  struct Points { std::vector<double> x, y, z; };

  std::vector<double> interleave (const Points& p)
  {
    std::vector<double> xyz(3*p.x.size());
    for (size_t i=0; i<p.x.size(); ++i) {
      xyz[3*i] = p.x[i];
      xyz[3*i+1] = p.y[i];
      xyz[3*i+2] = p.z[i];
    }
    return xyz;
  }

  Points uniform (std::mt19937_64& rng, const size_t n, const double lo, const double hi)
  {
    std::uniform_real_distribution<double> u(lo, hi);
    Points p;
    for (size_t i=0; i<n; ++i) { p.x.push_back(u(rng)); p.y.push_back(u(rng)); p.z.push_back(u(rng)); }
    return p;
  }

  // A few tight clusters over a uniform background, clipped to [0, 1].
  Points clustered (std::mt19937_64& rng, const size_t n)
  {
    std::uniform_real_distribution<double> u(0., 1.);
    std::normal_distribution<double> g(0., 0.01);
    double centres[5][3];
    for (auto& c : centres) for (double& v : c) v = u(rng);
    Points p;
    for (size_t i=0; i<n; ++i) {
      double q[3];
      const bool background = u(rng) < 0.1;
      const auto& c = centres[i%5];
      for (int a=0; a<3; ++a) q[a] = std::min(1., std::max(0., background ? u(rng) : c[a]+g(rng)));
      p.x.push_back(q[0]); p.y.push_back(q[1]); p.z.push_back(q[2]);
    }
    return p;
  }

  // Many objects on a handful of distinct points, among them -0.0 and +0.0 on
  // every axis: std::minmax_element in the original build kept the first
  // smallest and the last largest, and which zero that is decides the sign of
  // a fitted box edge. Every form must pick the same one.
  Points coincident (std::mt19937_64& rng, const size_t n)
  {
    const double values[] = {-0.0, 0.0, 0.25, 0.5, 1.0};
    std::uniform_int_distribution<int> pick(0, 4);
    Points p;
    for (size_t i=0; i<n; ++i) {
      p.x.push_back(values[pick(rng)]); p.y.push_back(values[pick(rng)]); p.z.push_back(values[pick(rng)]);
    }
    return p;
  }

  bool sameBits (const double a, const double b)
  {
    return std::memcmp(&a, &b, sizeof a) == 0;
  }

  void checkSameGeometry (const MeshGrid& a, const MeshGrid& b)
  {
    const Lims la = a.get_lims(), lb = b.get_lims();
    for (int i=0; i<3; ++i) for (int j=0; j<2; ++j) assert(sameBits(la[i][j], lb[i][j]));
    assert(a.get_nCells() == b.get_nCells());
    assert(sameBits(a.get_cellsize(), b.get_cellsize()));
    assert(a.get_periodic() == b.get_periodic());
    assert(a.get_nObjects() == b.get_nObjects());
  }

  // Every cell, through its centre: the objects and their order within it.
  void checkSameCells (const MeshGrid& a, const MeshGrid& b)
  {
    const Lims l = a.get_lims();
    const auto n = a.get_nCells();
    const double c = a.get_cellsize();
    for (unsigned int i=0; i<n[0]; ++i)
      for (unsigned int j=0; j<n[1]; ++j)
        for (unsigned int k=0; k<n[2]; ++k) {
          const double x = std::min(l[0][1], l[0][0]+(i+0.5)*c);
          const double y = std::min(l[1][1], l[1][0]+(j+0.5)*c);
          const double z = std::min(l[2][1], l[2][0]+(k+0.5)*c);
          assert(a.get_objectsInCell(x, y, z) == b.get_objectsInCell(x, y, z));
        }
  }

  void checkSameAnswers (const MeshGrid& a, const MeshGrid& b, std::mt19937_64& rng, const size_t queries)
  {
    const Lims l = a.get_lims();
    std::uniform_real_distribution<double> u(0., 1.);
    Points q;
    for (size_t i=0; i<queries; ++i) {
      q.x.push_back(l[0][0]+u(rng)*(l[0][1]-l[0][0]));
      q.y.push_back(l[1][0]+u(rng)*(l[1][1]-l[1][0]));
      q.z.push_back(l[2][0]+u(rng)*(l[2][1]-l[2][0]));
    }
    std::vector<unsigned int> idx;
    for (unsigned int i=0; i<a.get_nObjects() && idx.size() < queries; i += 1+i%3)
      if (a.isAlive(i)) idx.push_back(i);

    const double c = a.get_cellsize();
    const double radii[][2] = {{0.5*c, 0.}, {1.5*c, 0.3*c}, {4.*c, 0.}, {std::numeric_limits<double>::infinity(), 2.*c}};
    const unsigned int N = std::min(7u, a.get_nObjects()-1);

    for (size_t i=0; i<queries; ++i) {
      assert(a.nearestObject(q.x[i], q.y[i], q.z[i]) == b.nearestObject(q.x[i], q.y[i], q.z[i]));
      assert(a.nearestObjects(N, q.x[i], q.y[i], q.z[i]) == b.nearestObjects(N, q.x[i], q.y[i], q.z[i]));
      for (const auto& r : radii) {
        assert(a.closeObjects(q.x[i], q.y[i], q.z[i], r[0], r[1]) == b.closeObjects(q.x[i], q.y[i], q.z[i], r[0], r[1]));
        assert(a.countObjects(q.x[i], q.y[i], q.z[i], r[0], r[1]) == b.countObjects(q.x[i], q.y[i], q.z[i], r[0], r[1]));
      }
    }
    for (const unsigned int i : idx) {
      assert(a.nearestObject(i) == b.nearestObject(i));
      assert(a.nearestObjects(N, i) == b.nearestObjects(N, i));
      for (const auto& r : radii) {
        assert(a.closeObjects(i, r[0], r[1]) == b.closeObjects(i, r[0], r[1]));
        assert(a.countObjects(i, r[0], r[1]) == b.countObjects(i, r[0], r[1]));
      }
    }

    assert(a.nearestObject(q.x, q.y, q.z) == b.nearestObject(q.x, q.y, q.z));
    assert(a.nearestObjects(N, q.x, q.y, q.z) == b.nearestObjects(N, q.x, q.y, q.z));
    assert(a.nearestObject(idx) == b.nearestObject(idx));
    assert(a.nearestObjects(N, idx) == b.nearestObjects(N, idx));
    for (const auto& r : radii) {
      const MeshGrid::Batch pa = a.closeObjects(q.x, q.y, q.z, r[0], r[1]), pb = b.closeObjects(q.x, q.y, q.z, r[0], r[1]);
      assert(pa.indices == pb.indices && pa.offsets == pb.offsets);
      const MeshGrid::Batch ia = a.closeObjects(idx, r[0], r[1]), ib = b.closeObjects(idx, r[0], r[1]);
      assert(ia.indices == ib.indices && ia.offsets == ib.offsets);
      assert(a.countObjects(q.x, q.y, q.z, r[0], r[1]) == b.countObjects(q.x, q.y, q.z, r[0], r[1]));
      assert(a.countObjects(idx, r[0], r[1]) == b.countObjects(idx, r[0], r[1]));
    }
  }

  void checkSame (const MeshGrid& a, const MeshGrid& b, std::mt19937_64& rng, const size_t queries)
  {
    checkSameGeometry(a, b);
    checkSameCells(a, b);
    checkSameAnswers(a, b, rng, queries);
  }

  // Builds the grid with all three forms (and, for default options, all
  // three no-options overloads too), overwrites the caller's arrays, and
  // checks that the pointer grids match the vector grid, before and after
  // the same sequence of removals and additions.
  void checkForms (const Points& p, const double cellsize, const std::optional<MeshGrid::Options>& options,
                   std::mt19937_64& rng, const size_t queries)
  {
    Points copy = p;
    std::vector<double> xyz = interleave(p);
    const size_t n = p.x.size();

    std::vector<MeshGrid> grids;
    if (options) {
      grids.emplace_back(p.x, p.y, p.z, cellsize, *options);
      grids.emplace_back(copy.x.data(), copy.y.data(), copy.z.data(), n, cellsize, *options);
      grids.emplace_back(xyz.data(), n, cellsize, *options);
    }
    else {
      grids.emplace_back(p.x, p.y, p.z, cellsize);
      grids.emplace_back(copy.x.data(), copy.y.data(), copy.z.data(), n, cellsize);
      grids.emplace_back(xyz.data(), n, cellsize);
      grids.emplace_back(copy.x.data(), copy.y.data(), copy.z.data(), n, cellsize, MeshGrid::Options());
      grids.emplace_back(xyz.data(), n, cellsize, MeshGrid::Options());
    }

    // Overwrite, then release, the arrays the pointer grids were built from.
    for (double& v : copy.x) v = std::numeric_limits<double>::quiet_NaN();
    for (double& v : xyz) v = std::numeric_limits<double>::quiet_NaN();
    copy = Points();
    xyz = std::vector<double>();

    for (size_t g=1; g<grids.size(); ++g) checkSame(grids[0], grids[g], rng, queries);

    // The same edits on every grid keep them equal.
    const Lims l = grids[0].get_lims();
    std::uniform_real_distribution<double> u(0., 1.);
    std::vector<unsigned int> removed;
    for (unsigned int i=0; i<grids[0].get_nObjects(); i += 5) removed.push_back(i);
    std::vector<std::array<double,3>> added;
    for (int k=0; k<10; ++k)
      added.push_back({l[0][0]+u(rng)*(l[0][1]-l[0][0]), l[1][0]+u(rng)*(l[1][1]-l[1][0]), l[2][0]+u(rng)*(l[2][1]-l[2][0])});
    for (auto& g : grids) {
      for (const unsigned int i : removed) g.removeObject(i);
      for (const auto& a : added) g.addObject(a[0], a[1], a[2]);
    }
    for (size_t g=1; g<grids.size(); ++g) checkSame(grids[0], grids[g], rng, queries/2);
  }

  std::string errorOf (const std::function<void()>& build)
  {
    try { build(); }
    catch (const meshsearch::IndexError&) { return "IndexError"; }
    catch (const meshsearch::Error& e) { return e.what(); }
    return "";
  }

  // Each invalid input raises the vector form's Error, with its message,
  // through every form.
  void checkSameError (const Points& p, const double cellsize, const MeshGrid::Options& opt)
  {
    const std::vector<double> xyz = interleave(p);
    const size_t n = p.x.size();
    const std::string expected = errorOf([&] { MeshGrid(p.x, p.y, p.z, cellsize, opt); });
    assert(!expected.empty() && expected != "IndexError");
    assert(errorOf([&] { MeshGrid(p.x.data(), p.y.data(), p.z.data(), n, cellsize, opt); }) == expected);
    assert(errorOf([&] { MeshGrid(xyz.data(), n, cellsize, opt); }) == expected);
  }

  void testErrors ()
  {
    std::mt19937_64 rng(7);
    const Points good = uniform(rng, 50, 0., 1.);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    MeshGrid::Options none;
    MeshGrid::Options box;
    box.limits = Lims{{{0., 1.}, {0., 1.}, {0., 1.}}};

    // A non-finite coordinate is named by its axis and its point, the
    // first one found axis by axis; in the interleaved array it is still
    // point k, not element 3k+a.
    for (int axis=0; axis<3; ++axis)
      for (const double bad : {nan, inf, -inf}) {
        Points p = good;
        std::vector<double>* axes[3] = {&p.x, &p.y, &p.z};
        (*axes[axis])[17] = bad;
        (*axes[2-axis])[31] = bad;
        checkSameError(p, 0.2, none);
        checkSameError(p, 0.2, box);
      }

    // A coordinate outside the limits, on each axis and at each end.
    for (int axis=0; axis<3; ++axis)
      for (const double bad : {-0.5, 1.5}) {
        Points p = good;
        std::vector<double>* axes[3] = {&p.x, &p.y, &p.z};
        (*axes[axis])[9] = bad;
        checkSameError(p, 0.2, box);
      }

    // cellsize not finite and positive, or with an overflowing reciprocal.
    for (const double c : {0., -1., nan, inf, 1e-320}) {
      checkSameError(good, c, none);
      checkSameError(good, c, box);
    }

    // Limits not finite or inverted; periodic without limits; periods that
    // are not integer multiples of each other.
    MeshGrid::Options bad = box;
    (*bad.limits)[1] = {1., 0.};
    checkSameError(good, 0.2, bad);
    (*bad.limits)[1] = {0., inf};
    checkSameError(good, 0.2, bad);
    MeshGrid::Options periodic;
    periodic.periodic = {true, false, false};
    checkSameError(good, 0.2, periodic);
    MeshGrid::Options uneven = box;
    uneven.periodic = {true, true, true};
    (*uneven.limits)[2] = {0., 1.5};
    Points q = good;
    for (double& v : q.z) v *= 1.5;
    checkSameError(q, 0.2, uneven);

    // No points and no limits.
    checkSameError(Points(), 0.2, none);

    // A null pointer is an error when there are points to read.
    assert(errorOf([&] { MeshGrid(nullptr, good.y.data(), good.z.data(), 50, 0.2, box); }) == "a coordinate array is a null pointer");
    assert(errorOf([&] { MeshGrid(good.x.data(), nullptr, good.z.data(), 50, 0.2); }) == "a coordinate array is a null pointer");
    assert(errorOf([&] { MeshGrid(good.x.data(), good.y.data(), nullptr, 50, 0.2); }) == "a coordinate array is a null pointer");
    assert(errorOf([&] { MeshGrid(nullptr, 50, 0.2, box); }) == "the coordinate array is a null pointer");
    assert(errorOf([&] { MeshGrid(nullptr, 50, 0.2); }) == "the coordinate array is a null pointer");

    // With no points, null pointers are fine: limits give an empty grid, and
    // without them the error is the vector form's.
    const MeshGrid e1(nullptr, nullptr, nullptr, 0, 0.2, box);
    const MeshGrid e2(nullptr, 0, 0.2, box);
    const MeshGrid e0(std::vector<double>(), std::vector<double>(), std::vector<double>(), 0.2, box);
    assert(e1.get_nObjects() == 0 && e2.get_nObjects() == 0);
    checkSameGeometry(e0, e1);
    checkSameGeometry(e0, e2);
    const std::string empty = errorOf([&] { MeshGrid(std::vector<double>(), std::vector<double>(), std::vector<double>(), 0.2); });
    assert(errorOf([&] { MeshGrid(nullptr, nullptr, nullptr, 0, 0.2); }) == empty);
    assert(errorOf([&] { MeshGrid(nullptr, 0, 0.2); }) == empty);

    // More points than indices can name: refused before anything is read,
    // so the pointers need not be valid for that many.
    const size_t too_many = size_t(std::numeric_limits<unsigned int>::max())+1;
    const std::string many = "too many objects: at most 4294967295 are supported";
    assert(errorOf([&] { MeshGrid(good.x.data(), good.y.data(), good.z.data(), too_many, 0.2, box); }) == many);
    assert(errorOf([&] { MeshGrid(good.x.data(), too_many, 0.2, box); }) == many);
  }

}

int main ()
{
  std::mt19937_64 rng(20261127);

  // Dense (about 130 objects per cell), about two per cell, and sparse (about
  // one object per 15 cells) uniform data, on a fitted box and on explicit limits wider than the data.
  MeshGrid::Options wide;
  wide.limits = Lims{{{-0.25, 1.5}, {-1., 1.25}, {0., 1.}}};
  for (const double cellsize : {0.1, 0.4}) {
    const Points p = uniform(rng, 2000, 0., 1.);
    checkForms(p, cellsize, std::nullopt, rng, 150);
    checkForms(p, cellsize, wide, rng, 150);
  }
  checkForms(uniform(rng, 400, 0., 1.), 0.055, std::nullopt, rng, 100);

  // Clustered data, and a box far from the origin.
  checkForms(clustered(rng, 3000), 0.05, std::nullopt, rng, 150);
  checkForms(uniform(rng, 500, 1e6, 1e6+1.), 0.2, std::nullopt, rng, 100);

  // Coincident points and signed zeros.
  checkForms(coincident(rng, 600), 0.2, std::nullopt, rng, 100);
  checkForms(coincident(rng, 600), 0.2, wide, rng, 100);

  // Periodic and mixed boxes.
  MeshGrid::Options cube;
  cube.limits = Lims{{{0., 1.}, {0., 1.}, {0., 1.}}};
  cube.periodic = {true, true, true};
  checkForms(uniform(rng, 1500, 0., 1.), 0.15, cube, rng, 150);
  MeshGrid::Options mixed = cube;
  mixed.periodic = {true, false, true};
  checkForms(clustered(rng, 1500), 0.15, mixed, rng, 150);

  // Three points, and points on a plane (an axis of zero extent).
  checkForms(uniform(rng, 3, 0., 1.), 0.2, cube, rng, 20);
  Points flat = uniform(rng, 300, 0., 1.);
  for (double& z : flat.z) z = 0.5;
  checkForms(flat, 0.1, std::nullopt, rng, 100);

  testErrors();

  std::puts("test_constructors: all checks passed");
  return 0;
}
