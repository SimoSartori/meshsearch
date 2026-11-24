// Tests for the interface guarantees of meshsearch::MeshGrid that concern the
// object rather than a query: what the type is, how it moves, and the members
// that ask about indices and points without running a query.
//
//   - The class is final, copies and moves are explicitly defaulted and a
//     move is a real move: it allocates nothing.
//   - get_lims returns the type of Options::limits, so a box round-trips.
//   - isAlive answers false, and does not raise, on an index that is out of
//     range or removed; its batch does the same for every entry.
//   - contains(X, Y, Z) is true exactly when a query at that point is
//     accepted, in the dense and the sparse regime, on periodic and
//     non-periodic grids, for points inside, on the edge, outside, NaN and
//     infinite. Its batch agrees point by point and checks the lengths.
//
// The allocation counter replaces global operator new for this program only.
//
// NDEBUG is undefined below so that assert() stays live whatever the build
// type.

#undef NDEBUG

#include "meshsearch/MeshGrid.h"
#include "meshsearch/Version.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <new>
#include <random>
#include <type_traits>
#include <vector>

namespace {
  long g_allocations = 0;
}

void* operator new (std::size_t n)
{
  ++g_allocations;
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void operator delete (void* p) noexcept { std::free(p); }
void operator delete (void* p, std::size_t) noexcept { std::free(p); }

using meshsearch::MeshGrid;
using Lims = std::array<std::array<double,2>,3>;

static_assert(std::is_final_v<MeshGrid>);
static_assert(std::is_copy_constructible_v<MeshGrid> && std::is_copy_assignable_v<MeshGrid>);
static_assert(std::is_nothrow_move_constructible_v<MeshGrid>);
static_assert(std::is_nothrow_move_assignable_v<MeshGrid>);
static_assert(std::is_same_v<decltype(std::declval<const MeshGrid&>().get_lims()), Lims>);
static_assert(std::is_same_v<decltype(*std::declval<MeshGrid::Options&>().limits), Lims&>);
static_assert(std::is_same_v<decltype(std::declval<const MeshGrid&>().contains(0., 0., 0.)), bool>);
static_assert(std::is_same_v<decltype(std::declval<const MeshGrid&>().isAlive(std::vector<unsigned int>{})),
                             std::vector<std::uint8_t>>);

namespace {

  struct Points { std::vector<double> x, y, z; };

  Points random_points (std::mt19937_64& rng, const size_t n, const double lo, const double hi)
  {
    std::uniform_real_distribution<double> u(lo, hi);
    Points p;
    for (size_t i=0; i<n; ++i) { p.x.push_back(u(rng)); p.y.push_back(u(rng)); p.z.push_back(u(rng)); }
    return p;
  }

  bool accepted (const MeshGrid& g, const double x, const double y, const double z)
  {
    try { g.nearestObject(x, y, z); }
    catch (const meshsearch::IndexError&) { return false; }
    catch (const meshsearch::Error&) { return false; }
    return true;
  }

  void test_type_and_moves ()
  {
    const Points p = random_points(*std::make_unique<std::mt19937_64>(1), 2000, 0., 1.);
    MeshGrid a(p.x, p.y, p.z, 0.1);
    const auto nearest = a.nearestObject(0.5, 0.5, 0.5);
    const auto box = a.get_lims();

    const long before = g_allocations;
    MeshGrid b(std::move(a));
    MeshGrid c;
    c = std::move(b);
    assert(g_allocations == before);
    assert(c.nearestObject(0.5, 0.5, 0.5) == nearest);
    assert(c.get_lims() == box);

    MeshGrid d(c);
    assert(g_allocations > before);
    assert(d.nearestObject(0.5, 0.5, 0.5) == nearest);

    MeshGrid::Options opt;
    opt.limits = c.get_lims();
    MeshGrid again(p.x, p.y, p.z, c.get_cellsize(), opt);
    assert(again.get_lims() == box);

    assert(std::string(MESHSEARCH_VERSION).size() > 4);
    assert(MESHSEARCH_VERSION_MAJOR >= 0);
  }

  void test_is_alive ()
  {
    MeshGrid g({0.1, 0.5, 0.9}, {0.1, 0.5, 0.9}, {0.1, 0.5, 0.9}, 0.25);
    g.removeObject(1);
    const unsigned int added = g.addObject(0.3, 0.3, 0.3);
    assert(added == 3);

    assert(g.isAlive(0) && !g.isAlive(1) && g.isAlive(2) && g.isAlive(3));
    assert(!g.isAlive(4) && !g.isAlive(0xFFFFFFFFu));

    const std::vector<unsigned int> idx = {3, 1, 0, 99, 2, 1, 0xFFFFFFFFu};
    const std::vector<std::uint8_t> expected = {1, 0, 1, 0, 1, 0, 0};
    assert(g.isAlive(idx) == expected);

    std::vector<std::uint8_t> out(100, 7);
    const auto cap = out.capacity();
    g.isAlive(idx, out);
    assert(out == expected && out.capacity() == cap);
    g.isAlive({}, out);
    assert(out.empty());

    MeshGrid none;
    assert(!none.isAlive(0));
  }

  void check_contains (const MeshGrid& g, const Points& q)
  {
    const auto flags = g.contains(q.x, q.y, q.z);
    assert(flags.size() == q.x.size());
    std::vector<std::uint8_t> out(3, 9);
    g.contains(q.x, q.y, q.z, out);
    assert(out == flags);
    for (size_t i=0; i<q.x.size(); ++i) {
      const bool c = g.contains(q.x[i], q.y[i], q.z[i]);
      assert(flags[i] == (c ? 1 : 0));
      assert(c == accepted(g, q.x[i], q.y[i], q.z[i]));
    }
  }

  void test_contains ()
  {
    std::mt19937_64 rng(7);
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    for (const size_t n : {size_t(20000), size_t(30)}) {
      const Points p = random_points(rng, n, 0., 1.);
      const double cellsize = n > 1000 ? 0.05 : 0.4;
      MeshGrid::Options plain;
      MeshGrid::Options fixed;
      fixed.limits = Lims{{{0., 1.}, {0., 1.}, {0., 1.}}};
      MeshGrid::Options periodic = fixed;
      periodic.periodic = {true, false, true};
      for (const auto& opt : {plain, fixed, periodic}) {
        const MeshGrid g(p.x, p.y, p.z, cellsize, opt);
        const auto box = g.get_lims();
        Points q = random_points(rng, 3000, -0.5, 1.5);
        for (int a=0; a<3; ++a) {
          for (const double edge : {box[a][0], box[a][1],
                                    std::nextafter(box[a][0], -inf), std::nextafter(box[a][1], inf),
                                    inf, -inf, nan}) {
            Points e = random_points(rng, 1, 0.2, 0.8);
            (a == 0 ? e.x : a == 1 ? e.y : e.z)[0] = edge;
            q.x.push_back(e.x[0]); q.y.push_back(e.y[0]); q.z.push_back(e.z[0]);
          }
        }
        check_contains(g, q);
      }
    }

    const MeshGrid g({0.1, 0.5}, {0.1, 0.5}, {0.1, 0.5}, 0.25);
    bool threw = false;
    try { g.contains({0.1, 0.2}, {0.1}, {0.1, 0.2}); } catch (const meshsearch::Error&) { threw = true; }
    assert(threw);
    // Braced empty arguments would pick the scalar overload, so the vectors are named.
    const std::vector<double> nothing;
    assert(g.contains(nothing, nothing, nothing).empty());

    const MeshGrid none;
    assert(!none.contains(0., 0., 0.));
  }

}

int main ()
{
  test_type_and_moves();
  test_is_alive();
  test_contains();
  std::printf("test_api: ok\n");
  return 0;
}
