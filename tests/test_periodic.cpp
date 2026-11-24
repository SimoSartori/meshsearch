// Tests for the periodic axes of meshsearch::MeshGrid. They check the
// guarantees stated in include/meshsearch/MeshGrid.h for a grid with a
// periodic axis: every query returns what the specification in the header
// says, and nothing depends on how objects fall into cells.
//
// The load-bearing groups compare every query against a brute force that
// implements the header's definition of the distance literally. It takes the
// displacement y-q on each axis, replaces it by (y-q)-L if it is >= L/2 and by
// (y-q)+L if it is < -L/2, and sums the squares as dx*dx+dy*dy+dz*dz. It must be
// compiled without floating-point contraction, like the library, or the
// reference would round differently from the grid, and the points where that
// matters are exactly the ones these tests are built around.
//
// Most of the data is hostile on purpose. The decision between an image and
// the next one is made on a rounded quantity, so a wrong cell rule shows only
// for objects within a few ulp of half a period from the query, of a cell
// face, or of an end of the box; and the error budget of the cell geometry
// depends on the size of the coordinates, so boxes far from the origin and
// cells a few ulp wide are included. Random data alone would pass a broken rule.
//
// NDEBUG is undefined below so that assert() stays live whatever the build
// type. MESHSEARCH_PERIODIC_SCALE (default 1) multiplies the number of
// configurations and queries. MESHSEARCH_COUNT_FAILURES, when set, makes a
// failed check count instead of abort, and the counts are printed per check;
// it is how a deliberately weakened library is shown to fail.

#undef NDEBUG

#include "meshsearch/MeshGrid.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

using meshsearch::Error;
using meshsearch::IndexError;
using meshsearch::MeshGrid;

namespace {

  using Box = std::array<std::array<double,2>,3>;

  const double inf = std::numeric_limits<double>::infinity();

  std::mt19937_64 rng(20261125);

  double uniform (const double a, const double b) { return std::uniform_real_distribution<double>(a, b)(rng); }
  int integer (const int a, const int b) { return std::uniform_int_distribution<int>(a, b)(rng); }
  bool coin (const double p = 0.5) { return uniform(0., 1.) < p; }

  double ulpOf (const double x)
  {
    const double a = std::fabs(x);
    return std::nextafter(a, inf)-a;
  }

  double stepUlps (double x, const int steps)
  {
    for (int i=0; i<std::abs(steps); ++i) x = std::nextafter(x, steps > 0 ? inf : -inf);
    return x;
  }

  bool countOnly = false;
  std::map<std::string, long> failed;

  void check (const bool ok, const char* const what)
  {
    if (ok) return;
    ++failed[what];
    if (countOnly) return;
    printf("FAILED: %s\n", what);
    std::abort();
  }

  int scale = 1;

  // The smallest R with R*R >= v, and the largest with R*R <= v. Radii built
  // from an object's own squared distance put the threshold exactly on that
  // object, which is where ties and the stop rules are tested.
  double radiusAbove (const double v)
  {
    double R = std::sqrt(v);
    while (R*R < v) R = std::nextafter(R, inf);
    for (;;) {
      const double P = std::nextafter(R, 0.);
      if (P != R && P*P >= v) R = P; else break;
    }
    return R;
  }

  double radiusBelow (const double v)
  {
    if (v == 0.) return 0.;
    double R = std::sqrt(v);
    while (R*R > v) R = std::nextafter(R, 0.);
    for (;;) {
      const double P = std::nextafter(R, inf);
      if (P != R && P*P <= v) R = P; else break;
    }
    return R;
  }

  // The header's definition of the displacement along one axis. A
  // non-periodic axis has half = inf, so neither correction applies.
  double image (const double d0, const double period, const double half)
  {
    if (d0 >= half) return d0-period;
    if (d0 < -half) return d0+period;
    return d0;
  }

  // The reference: the points the grid holds, with which are still alive.
  struct World {
    std::vector<double> c[3];
    std::vector<char> alive;
    Box lim;
    std::array<bool,3> per;
    double period[3], half[3];

    void finish ()
    {
      for (int a=0; a<3; ++a) {
        period[a] = per[a] ? lim[a][1]-lim[a][0] : 0.;
        half[a] = per[a] ? period[a]*0.5 : inf;
      }
      alive.assign(c[0].size(), 1);
    }

    size_t size () const { return c[0].size(); }

    void add (const double x, const double y, const double z)
    {
      c[0].push_back(x); c[1].push_back(y); c[2].push_back(z);
      alive.push_back(1);
    }

    double sq (const size_t i, const double* q) const
    {
      const double dx = image(c[0][i]-q[0], period[0], half[0]);
      const double dy = image(c[1][i]-q[1], period[1], half[1]);
      const double dz = image(c[2][i]-q[2], period[2], half[2]);
      return dx*dx+dy*dy+dz*dz;
    }

    unsigned int nAlive () const
    {
      return static_cast<unsigned int>(std::count(alive.begin(), alive.end(), char(1)));
    }
  };

  MeshGrid build (const World& w, const double cellsize)
  {
    MeshGrid::Options options;
    options.limits = w.lim;
    options.periodic = w.per;
    return MeshGrid(w.c[0], w.c[1], w.c[2], cellsize, options);
  }

  // The accessors of a periodic grid: the box is the limits exactly, every
  // cell is at most cellsize, and each periodic axis holds a whole number of
  // cells that spans its period within the ratio tolerance.
  void checkAccessors (const MeshGrid& g, const World& w, const double cellsize)
  {
    check(g.get_periodic() == w.per, "get_periodic");
    const auto lims = g.get_lims();
    const double c = g.get_cellsize();
    const auto n = g.get_nCells();
    check(c <= cellsize && c > 0., "cell side at most cellsize");
    for (int a=0; a<3; ++a) {
      if (!w.per[a]) continue;
      check(lims[a][0] == w.lim[a][0] && lims[a][1] == w.lim[a][1], "periodic box returned as given");
      check(std::fabs(double(n[a])*c-w.period[a]) <= 4.e-12*w.period[a], "periodic axis spans its period");
    }
  }

  // A coordinate on axis a chosen where a rule is most likely to be wrong:
  // the ends of the box and their neighbours, a cell face, half a period from
  // another coordinate. base is that other coordinate.
  double edgeCoordinate (const World& w, const int a, const double c, const double base)
  {
    const double lo = w.lim[a][0], hi = w.lim[a][1];
    double v;
    switch (integer(0, 5)) {
      case 0: v = lo; break;
      case 1: v = hi; break;
      case 2: v = std::nextafter(lo, inf); break;
      case 3: v = std::nextafter(hi, -inf); break;
      case 4: v = lo+c*double(integer(0, 12)); break;
      default:
        v = w.per[a] ? (coin() ? base+w.half[a] : base-w.half[a]) : base;
        break;
    }
    v = stepUlps(v, integer(-4, 4));
    if (v < lo) v = coin() ? lo : v+(w.per[a] ? w.period[a] : 0.);
    if (v > hi) v = coin() ? hi : v-(w.per[a] ? w.period[a] : 0.);
    return std::min(std::max(v, lo), hi);
  }

  // One query point, and the object it is taken from if it is the stored
  // position of one (self, else -1).
  void makeQuery (const World& w, const double c, double* q, long& self)
  {
    self = -1;
    const int kind = integer(0, 9);
    long s = -1;
    if (kind >= 3) {
      do s = integer(0, int(w.size())-1); while (!w.alive[s]);
    }
    if (kind < 3) {
      for (int a=0; a<3; ++a) q[a] = uniform(w.lim[a][0], w.lim[a][1]);
    }
    else if (kind < 6) {
      for (int a=0; a<3; ++a) q[a] = w.c[a][s];
      self = s;
    }
    else {
      for (int a=0; a<3; ++a) q[a] = coin(0.7) ? edgeCoordinate(w, a, c, w.c[a][s]) : w.c[a][s];
    }
  }

  // Every query kind against the reference, at one point.
  void checkQuery (const MeshGrid& g, const World& w, const double* q, const long self, const double scaleLength)
  {
    std::vector<unsigned int> order;
    std::vector<double> d(w.size(), 0.);
    for (size_t i=0; i<w.size(); ++i) {
      if (!w.alive[i]) continue;
      d[i] = w.sq(i, q);
      order.push_back(static_cast<unsigned int>(i));
    }
    std::sort(order.begin(), order.end(), [&] (const unsigned int a, const unsigned int b) {
      return d[a] < d[b] || (d[a] == d[b] && a < b);
    });

    for (const bool byIndex : {false, true}) {
      if (byIndex && self < 0) continue;
      std::vector<unsigned int> ref;
      for (const unsigned int i : order)
        if (!byIndex || long(i) != self) ref.push_back(i);
      const unsigned int nRef = static_cast<unsigned int>(ref.size());

      const unsigned int me = static_cast<unsigned int>(self);
      auto close = [&] (const double Rmax, const double Rmin) {
        return byIndex ? g.closeObjects(me, Rmax, Rmin) : g.closeObjects(q[0], q[1], q[2], Rmax, Rmin);
      };

      if (nRef > 0) {
        const unsigned int r = byIndex ? g.nearestObject(me) : g.nearestObject(q[0], q[1], q[2]);
        check(r < w.size() && w.alive[r] && d[r] == d[ref[0]] && !(byIndex && long(r) == self), "nearestObject");
      }

      for (const unsigned int N : {1u, 2u, 8u, 56u, 57u, 113u, nRef}) {
        if (N > nRef) continue;
        const auto got = byIndex ? g.nearestObjects(N, me) : g.nearestObjects(N, q[0], q[1], q[2]);
        check(got == std::vector<unsigned int>(ref.begin(), ref.begin()+N), "nearestObjects");
      }

      std::vector<std::array<double,2>> radii;
      const double Lc[] = {0., 1.e-9*scaleLength, 0.5*g.get_cellsize(), 2.*g.get_cellsize(), 0.49*scaleLength,
                           0.5*scaleLength, scaleLength, 2.*scaleLength, inf};
      for (int k=0; k<3; ++k) {
        const double Rmax = Lc[integer(0, 8)];
        const double lowers[] = {0., Rmax, 0.5*Rmax};
        radii.push_back({Rmax, lowers[integer(0, 2)]});
      }
      if (nRef > 0) {
        const double v = d[ref[size_t(integer(0, int(nRef)-1))]];
        radii.push_back({radiusAbove(v), 0.});
        radii.push_back({radiusAbove(v), radiusBelow(v)});
        radii.push_back({inf, radiusBelow(v)});
        radii.push_back({radiusAbove(v), 0.5*radiusAbove(v)});
      }
      radii.push_back({inf, 0.});

      for (const auto& pair : radii) {
        const double Rmax_sq = pair[0]*pair[0], Rmin_sq = pair[1]*pair[1];
        std::vector<unsigned int> want;
        for (const unsigned int i : ref)
          if (d[i] <= Rmax_sq && d[i] >= Rmin_sq) want.push_back(i);
        auto got = close(pair[0], pair[1]);
        std::sort(got.begin(), got.end());
        std::sort(want.begin(), want.end());
        check(got == want, "closeObjects");
        if (pair[0] == inf && pair[1] == 0.) check(got.size() == nRef, "closeObjects(inf) returns every object once");
      }
    }
  }

  void checkWorld (const MeshGrid& g, const World& w, const double c, const int nQueries, const double scaleLength)
  {
    for (int k=0; k<nQueries; ++k) {
      double q[3];
      long self;
      makeQuery(w, c, q, self);
      checkQuery(g, w, q, self, scaleLength);
    }
  }

  double shortestPeriod (const World& w)
  {
    double L = inf;
    for (int a=0; a<3; ++a)
      if (w.per[a]) L = std::min(L, w.period[a]);
    return L;
  }

  std::array<bool,3> subset (const int mask)
  {
    return {bool(mask & 1), bool(mask & 2), bool(mask & 4)};
  }

  // Multiples of the shortest period for the periodic axes: up to three, with
  // one axis at exactly one, since {2, 3} would not be multiples of the
  // shortest. Non-periodic axes get 1.
  std::array<int,3> multiples (const std::array<bool,3>& per)
  {
    const int most = integer(1, 3);
    std::array<int,3> k = {1, 1, 1};
    int first = -1;
    for (int a=0; a<3; ++a) {
      if (!per[a]) continue;
      k[a] = integer(1, most);
      if (first < 0 || coin(0.4)) first = a;
    }
    if (first >= 0) k[first] = 1;
    return k;
  }

  // Random points, in dense (0.3 per cell), matched (3) and sparse (30)
  // regimes, for every non-empty set of periodic axes, with periods that are
  // one to three times the shortest. Non-periodic axes keep their box.
  void testUniform ()
  {
    for (int rep=0; rep<2*scale; ++rep)
      for (int mask=1; mask<8; ++mask)
        for (const int n : {200, 2000})
          for (const double perCell : {0.3, 3., 30.}) {
            World w;
            w.per = subset(mask);
            const double shortest = uniform(0.7, 2.);
            const auto k = multiples(w.per);
            double volume = 1.;
            for (int a=0; a<3; ++a) {
              const double lo = coin() ? 0. : uniform(-4., 4.);
              const double length = w.per[a] ? shortest*k[a] : uniform(1., 3.);
              w.lim[a] = {lo, lo+length};
              volume *= length;
            }
            w.finish();
            for (int i=0; i<n; ++i) w.add(uniform(w.lim[0][0], w.lim[0][1]), uniform(w.lim[1][0], w.lim[1][1]),
                                          uniform(w.lim[2][0], w.lim[2][1]));
            const double cellsize = std::cbrt(perCell*volume/n);
            const MeshGrid g = build(w, cellsize);
            checkAccessors(g, w, cellsize);
            checkWorld(g, w, g.get_cellsize(), n == 200 ? 600 : 300, shortestPeriod(w));
          }
    printf("  periodic grids of random points, dense to sparse, every axis subset: ok\n");
  }

  // Integer coordinates and integer periods, so that many squared distances
  // tie exactly, including at exactly half a period, for even and odd periods.
  void testLattice ()
  {
    for (int rep=0; rep<3*scale; ++rep)
      for (int mask=1; mask<8; ++mask) {
        World w;
        w.per = subset(mask);
        const int base = integer(0, 3) < 2 ? 6 : (coin() ? 7 : 12);
        const int k[3] = {integer(1, 2), integer(1, 2), integer(1, 2)};
        int length[3];
        for (int a=0; a<3; ++a) {
          length[a] = w.per[a] ? base*k[a] : integer(4, 8);
          w.lim[a] = {0., double(length[a])};
        }
        w.finish();
        for (int x=0; x<=length[0]; ++x)
          for (int y=0; y<=length[1]; ++y)
            for (int z=0; z<=length[2]; ++z)
              if (coin(0.25)) w.add(x, y, z);
        const double cellsizes[] = {1., 1.5, 2.5, 0.7, 3.};
        const double cellsize = cellsizes[integer(0, 4)];
        const MeshGrid g = build(w, cellsize);
        checkAccessors(g, w, cellsize);
        checkWorld(g, w, g.get_cellsize(), 400, shortestPeriod(w));
      }
    printf("  periodic lattices, ties at exactly half a period: ok\n");
  }

  // Objects and queries within a few ulp of half a period from each other, of
  // an end of the box and of a cell face, on grids of 1 to 12 cells per
  // period, with the cellsize at, just under and just over the exact ratio.
  void testHalfPeriod ()
  {
    for (int rep=0; rep<4*scale; ++rep)
      for (const int cells : {1, 2, 3, 4, 5, 7, 12})
        for (const double factor : {1., 0.999, 1.001}) {
          World w;
          w.per = subset(integer(1, 7));
          const double shortest = uniform(0.5, 3.);
          const auto k = multiples(w.per);
          for (int a=0; a<3; ++a) {
            const double lo = coin() ? 0. : uniform(-5., 5.);
            const double length = w.per[a] ? shortest*k[a] : uniform(1., 3.);
            w.lim[a] = {lo, lo+length};
          }
          w.finish();
          const double cellsize = shortest/cells*factor;
          for (int i=0; i<300; ++i) {
            double p[3];
            for (int a=0; a<3; ++a) {
              const double base = uniform(w.lim[a][0], w.lim[a][1]);
              p[a] = coin(0.6) ? edgeCoordinate(w, a, cellsize, base) : base;
            }
            w.add(p[0], p[1], p[2]);
          }
          const MeshGrid g = build(w, cellsize);
          checkAccessors(g, w, cellsize);
          checkWorld(g, w, g.get_cellsize(), 300, shortestPeriod(w));
        }
    printf("  objects and queries at half a period, box ends and cell faces: ok\n");
  }

  // Boxes far from the origin, with cells of 1 to 100 ulp of the coordinates.
  // The cell geometry then carries rounding of the order of a cell, so the
  // margins of the stop and skip rules are all that keeps results from being
  // lost. Coordinates are whole multiples of an ulp, which also makes
  // coincident objects and exact ties common.
  void testFarFromOrigin ()
  {
    for (int rep=0; rep<2*scale; ++rep)
      for (const double origin : {0., -7.3, 1.e3, -6.6e9, 3.e12})
        for (const int cells : {1, 2, 3, 4, 5, 7, 12}) {
          World w;
          w.per = subset(integer(1, 7));
          double step;
          if (origin == 0.) step = 0.37;
          else step = ulpOf(origin)*integer(1, 100);
          const double shortest = step*cells;
          const auto k = multiples(w.per);
          for (int a=0; a<3; ++a) {
            const double lo = origin;
            const double length = w.per[a] ? shortest*k[a] : shortest*integer(1, 3);
            w.lim[a] = {lo, lo+length};
          }
          w.finish();
          const double unit = origin == 0. ? 0.37/8 : ulpOf(origin);
          for (int i=0; i<120; ++i) {
            double p[3];
            for (int a=0; a<3; ++a) {
              const double span = w.lim[a][1]-w.lim[a][0];
              p[a] = std::min(w.lim[a][0]+unit*std::floor(uniform(0., span/unit+1.)), w.lim[a][1]);
              if (coin(0.3)) p[a] = edgeCoordinate(w, a, step, p[a]);
            }
            w.add(p[0], p[1], p[2]);
          }
          const MeshGrid g = build(w, step);
          checkAccessors(g, w, step);
          checkWorld(g, w, g.get_cellsize(), 200, shortestPeriod(w));
        }
    printf("  boxes far from the origin, cells down to one ulp wide: ok\n");
  }

  // Periods that are multiples of the shortest within 1e-12 are accepted and
  // answered correctly; a period 1.1e-12 off is refused. A period that is
  // not a whole multiple is refused.
  void testPeriodRatio ()
  {
    for (const double offset : {0., 1.e-13, -1.e-13, 9.9e-13, -9.9e-13}) {
      for (int rep=0; rep<3*scale; ++rep) {
        World w;
        w.per = {true, true, true};
        const double shortest = uniform(0.8, 2.);
        const int k[3] = {1, integer(1, 3), integer(1, 3)};
        for (int a=0; a<3; ++a) w.lim[a] = {0., shortest*k[a]*(1.+(a == 0 ? 0. : offset))};
        w.finish();
        for (int i=0; i<400; ++i) w.add(uniform(0., w.lim[0][1]), uniform(0., w.lim[1][1]), uniform(0., w.lim[2][1]));
        const double cellsize = shortest/integer(2, 6);
        const MeshGrid g = build(w, cellsize);
        checkAccessors(g, w, cellsize);
        checkWorld(g, w, g.get_cellsize(), 200, shortestPeriod(w));
      }
    }
    for (const double offset : {1.1e-12, -1.1e-12}) {
      World w;
      w.per = {true, true, true};
      w.lim = {{{0., 1.}, {0., 2.*(1.+offset)}, {0., 1.}}};
      w.finish();
      w.add(0.5, 0.5, 0.5);
      bool thrown = false;
      try { build(w, 0.25); } catch (const Error&) { thrown = true; }
      check(thrown, "period 1.1e-12 off a multiple is refused");
    }
    for (const double ratio : {1.5, 2.5, 1.0000001}) {
      World w;
      w.per = {true, true, false};
      w.lim = {{{0., 1.}, {0., ratio}, {0., 1.}}};
      w.finish();
      w.add(0.5, 0.5, 0.5);
      bool thrown = false;
      try { build(w, 0.25); } catch (const Error&) { thrown = true; }
      check(thrown, "period not a multiple is refused");
    }
    printf("  period ratio tolerance, accepted and refused: ok\n");
  }

  // One periodic axis of a million to ten million cells, the others one
  // cell, with a few objects. This is where the classification of a cell as
  // interior or end has the least room, and where a walk must cross millions
  // of empty cells to find an object.
  void testManyCells ()
  {
    const int counts[] = {1000000, 4000000, 10000000};
    for (int which=0; which<(scale > 1 ? 3 : 2); ++which) {
      World w;
      w.per = {true, false, false};
      const double period = uniform(1., 2.);
      w.lim = {{{0., period}, {0., 0.}, {0., 0.}}};
      w.finish();
      const int n = integer(10, 50);
      for (int i=0; i<n; ++i) w.add(coin(0.2) ? edgeCoordinate(w, 0, period/counts[which], uniform(0., period)) : uniform(0., period), 0., 0.);
      const double cellsize = period/counts[which];
      const MeshGrid g = build(w, cellsize);
      checkAccessors(g, w, cellsize);
      check(g.get_nCells()[0] == static_cast<unsigned int>(counts[which]), "cells along the long axis");
      checkWorld(g, w, g.get_cellsize(), 12, period);
    }
    printf("  a periodic axis of millions of cells: ok\n");
  }

  // Adding and removing objects, at the ends of the box and half a period
  // away, then every query again. Index stability and the box check are the
  // header's; the periodic distance has to follow the new objects.
  void testAddRemove ()
  {
    for (int rep=0; rep<6*scale; ++rep) {
      World w;
      w.per = subset(integer(1, 7));
      const double shortest = uniform(0.7, 2.);
      const auto k = multiples(w.per);
      for (int a=0; a<3; ++a) {
        const double lo = coin() ? 0. : uniform(-3., 3.);
        w.lim[a] = {lo, lo+(w.per[a] ? shortest*k[a] : uniform(1., 3.))};
      }
      w.finish();
      for (int i=0; i<150; ++i) w.add(uniform(w.lim[0][0], w.lim[0][1]), uniform(w.lim[1][0], w.lim[1][1]), uniform(w.lim[2][0], w.lim[2][1]));
      const double cellsize = shortest/integer(2, 6);
      MeshGrid g = build(w, cellsize);
      const double c = g.get_cellsize();

      for (int step=0; step<300; ++step) {
        if (coin(0.55)) {
          double p[3];
          for (int a=0; a<3; ++a) p[a] = coin(0.5) ? edgeCoordinate(w, a, c, uniform(w.lim[a][0], w.lim[a][1])) : uniform(w.lim[a][0], w.lim[a][1]);
          const unsigned int index = g.addObject(p[0], p[1], p[2]);
          check(index == w.size(), "addObject issues the next index");
          w.add(p[0], p[1], p[2]);
        }
        else if (w.nAlive() > 5) {
          unsigned int i;
          do i = static_cast<unsigned int>(integer(0, int(w.size())-1)); while (!w.alive[i]);
          g.removeObject(i);
          w.alive[i] = 0;
          bool thrown = false;
          try { g.nearestObject(i); } catch (const IndexError&) { thrown = true; }
          check(thrown, "a removed index raises");
        }
        if (step % 100 == 99) checkWorld(g, w, c, 150, shortestPeriod(w));
      }
      for (int a=0; a<3; ++a) {
        if (!w.per[a]) continue;
        double p[3] = {w.lim[0][0], w.lim[1][0], w.lim[2][0]};
        p[a] = std::nextafter(w.lim[a][1], inf);
        bool thrown = false;
        try { g.addObject(p[0], p[1], p[2]); } catch (const Error&) { thrown = true; }
        check(thrown, "a point outside a periodic axis is refused, not wrapped");
      }
    }
    printf("  adding and removing objects on periodic grids: ok\n");
  }

  // Moving every point by a vector, modulo the period, changes no distance
  // but the rounding of each. Results must agree except for objects within
  // 1e-9 (relative) of the threshold, whose side of it the rounding decides.
  // This checks the geometry independently of the reference above.
  void testOriginShift ()
  {
    for (int rep=0; rep<4*scale; ++rep) {
      World w;
      w.per = subset(integer(1, 7));
      const double shortest = uniform(0.7, 2.);
      const auto k = multiples(w.per);
      for (int a=0; a<3; ++a) {
        const double lo = coin() ? 0. : uniform(-3., 3.);
        w.lim[a] = {lo, lo+(w.per[a] ? shortest*k[a] : uniform(1., 3.))};
      }
      w.finish();
      for (int i=0; i<800; ++i) w.add(uniform(w.lim[0][0], w.lim[0][1]), uniform(w.lim[1][0], w.lim[1][1]), uniform(w.lim[2][0], w.lim[2][1]));
      double shift[3];
      for (int a=0; a<3; ++a) shift[a] = w.per[a] ? uniform(0., w.period[a]) : 0.;
      World v = w;
      auto move = [&] (const double y, const int a) {
        if (!w.per[a]) return y;
        const double lo = w.lim[a][0];
        return std::min(std::max(lo+std::fmod(y-lo+shift[a], w.period[a]), lo), w.lim[a][1]);
      };
      for (size_t i=0; i<w.size(); ++i)
        for (int a=0; a<3; ++a) v.c[a][i] = move(w.c[a][i], a);
      const double cellsize = shortest/integer(2, 6);
      const MeshGrid gw = build(w, cellsize), gv = build(v, cellsize);
      for (int k=0; k<120; ++k) {
        double q[3], p[3];
        for (int a=0; a<3; ++a) { q[a] = uniform(w.lim[a][0], w.lim[a][1]); p[a] = move(q[a], a); }
        const double R = uniform(0.05, 0.6)*shortest;
        auto a = gw.closeObjects(q[0], q[1], q[2], R), b = gv.closeObjects(p[0], p[1], p[2], R);
        std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
        std::vector<unsigned int> diff;
        std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(diff));
        for (const unsigned int i : diff)
          check(std::fabs(w.sq(i, q)-R*R) <= 1.e-9*R*R, "origin shift: close set");
        const unsigned int N = static_cast<unsigned int>(integer(1, 60));
        auto na = gw.nearestObjects(N, q[0], q[1], q[2]), nb = gv.nearestObjects(N, p[0], p[1], p[2]);
        const double cut = w.sq(na.back(), q);
        std::sort(na.begin(), na.end()); std::sort(nb.begin(), nb.end());
        diff.clear();
        std::set_symmetric_difference(na.begin(), na.end(), nb.begin(), nb.end(), std::back_inserter(diff));
        for (const unsigned int i : diff)
          check(std::fabs(w.sq(i, q)-cut) <= 1.e-9*cut, "origin shift: nearest set");
      }
    }
    printf("  a periodic grid and the same points shifted around the box: ok\n");
  }

  // Refusals and defaults of the constructor that only periodic axes have,
  // and the behaviour that must not change when no axis is periodic.
  void testConstructor ()
  {
    const std::vector<double> X = {0.5, 1.5}, Y = {0.5, 1.5}, Z = {0.5, 1.5};
    auto make = [&] (const Box* limits, const std::array<bool,3> per, const double cellsize = 0.5,
                     const std::vector<double>& x = {0.5, 1.5}) {
      MeshGrid::Options options;
      if (limits) options.limits = *limits;
      options.periodic = per;
      return MeshGrid(x, x, x, cellsize, options);
    };
    const Box unit = {{{0., 2.}, {0., 2.}, {0., 2.}}};
    auto refused = [&] (const char* what, auto&& f) {
      bool thrown = false;
      try { f(); } catch (const IndexError&) { thrown = false; } catch (const Error&) { thrown = true; }
      check(thrown, what);
    };

    for (int mask=1; mask<8; ++mask)
      refused("a periodic axis without limits", [&] { make(nullptr, subset(mask)); });
    refused("zero period", [&] { const Box b = {{{1., 1.}, {0., 2.}, {0., 2.}}}; make(&b, {true, false, false}, 0.5, {1., 1.}); });
    refused("period below 2^-1000", [&] { const Box b = {{{0., 0x1p-1001}, {0., 2.}, {0., 2.}}}; make(&b, {true, false, false}, 0.5, {0., 0.}); });
    refused("limits too large to shift", [&] { const Box b = {{{-0x1p999, 0x1p999}, {0., 2.}, {0., 2.}}}; make(&b, {true, false, false}, 1., {0., 0.}); });
    refused("period not a multiple", [&] { const Box b = {{{0., 2.}, {0., 3.}, {0., 2.}}}; make(&b, {true, true, false}); });
    refused("point outside a periodic axis", [&] { make(&unit, {true, true, true}, 0.5, {0.5, 2.5}); });
    refused("too many cells on a periodic axis", [&] { const Box b = {{{0., 1.}, {0., 3.}, {0., 1.}}}; make(&b, {true, true, false}, 1.e-9, {}); });
    refused("cell count of a long period times its multiple", [&] {
      const Box b = {{{0., 1.}, {0., 3.e9}, {0., 1.}}}; make(&b, {true, true, false}, 1., {}); });

    // A cellsize larger than the period gives one cell per period; the cell
    // is the period itself.
    {
      const MeshGrid g = make(&unit, {true, true, true}, 100.);
      assert(g.get_nCells() == (std::array<unsigned int,3>{1, 1, 1}));
      assert(g.get_cellsize() == 2.);
    }
    // Without a periodic axis the options change nothing: the box is grown
    // as before and the cellsize is the one given.
    {
      const MeshGrid a(X, Y, Z, 0.3), b(X, Y, Z, 0.3, MeshGrid::Options());
      assert(a.get_lims() == b.get_lims() && a.get_cellsize() == 0.3 && b.get_cellsize() == 0.3);
      assert(a.get_periodic() == (std::array<bool,3>{false, false, false}));
    }
    // The cell side never exceeds the cellsize, also where the quotient of
    // period and cellsize rounds down onto an integer.
    for (int i=0; i<2000; ++i) {
      const double period = uniform(0.5, 3.);
      const double cellsize = i%2 ? period/integer(1, 40) : std::nextafter(period/integer(1, 40), 0.);
      Box b = {{{0., period}, {0., period}, {0., period}}};
      const MeshGrid g = make(&b, {true, true, true}, cellsize, {0., period});
      check(g.get_cellsize() <= cellsize, "cell side at most cellsize");
    }
    printf("  constructor refusals and cell sides for periodic axes: ok\n");
  }

}


int main ()
{
  countOnly = std::getenv("MESHSEARCH_COUNT_FAILURES") != nullptr;
  if (const char* s = std::getenv("MESHSEARCH_PERIODIC_SCALE")) scale = std::max(1, std::atoi(s));

  printf("meshsearch::MeshGrid, periodic axes\n");

  testConstructor();
  testUniform();
  testLattice();
  testHalfPeriod();
  testFarFromOrigin();
  testPeriodRatio();
  testManyCells();
  testAddRemove();
  testOriginShift();

  if (countOnly) {
    long total = 0;
    for (const auto& entry : failed) { printf("  failed: %-60s %ld\n", entry.first.c_str(), entry.second); total += entry.second; }
    printf("failed checks: %ld\n", total);
    return 0;
  }

  printf("all tests passed\n");

  return 0;
}
