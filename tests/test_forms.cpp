// Tests for the container-first, counting and batch forms of
// meshsearch::MeshGrid. The header defines each of them through a single
// query that already exists and is tested against brute force elsewhere
// (test_meshgrid, test_periodic):
//
//   - a container-first form fills the caller's container, cleared first,
//     with what the returning form returns;
//   - countObjects returns the size of what closeObjects returns, with the
//     same arguments and the same errors;
//   - a batch returns, in input order, what the single query returns for each
//     query, in the same order within a query;
//   - a batch raises if any of its queries would, and checks the arguments
//     shared by all queries even when it holds no query;
//   - an index batch may write into the vector holding its indices.
//
// So these tests compare every new form with the single query, element by
// element and in order, rather than with a brute force. Equality of order is
// the point: closeObjects returns its indices in an unspecified order, and the
// batch is specified to return that same order for each query.
//
// NDEBUG is undefined below so that assert() stays live whatever the build
// type.

#undef NDEBUG

#include "meshsearch/MeshGrid.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>
#include <thread>
#include <type_traits>
#include <vector>

using meshsearch::Error;
using meshsearch::IndexError;
using meshsearch::MeshGrid;

namespace {

  using Box = std::array<std::array<double,2>,3>;
  using Indices = std::vector<unsigned int>;
  using Counts = std::vector<std::uint64_t>;

  const double inf = std::numeric_limits<double>::infinity();

  std::mt19937_64 rng(20261125);

  double uniform (const double a, const double b) { return std::uniform_real_distribution<double>(a, b)(rng); }
  int integer (const int a, const int b) { return std::uniform_int_distribution<int>(a, b)(rng); }

  // ------------------------------------------------------------------
  // Overload choice. The caller's container is always the last argument, and
  // a batch differs from the single query of the same name by its first
  // argument (a vector instead of a number). Each call a user can write must
  // reach the form it names, and a call with the wrong container must not
  // compile at all rather than reach some other overload. An ambiguous call
  // counts as not compiling here, so the first group also proves that none of
  // those calls is ambiguous. The first line is the call that, with the
  // container first, took a non-const vector of indices for an output.
  // ------------------------------------------------------------------

  template <class... A>
  using CloseCall = decltype(std::declval<const MeshGrid&>().closeObjects(std::declval<A>()...));
  template <class... A>
  using CountCall = decltype(std::declval<const MeshGrid&>().countObjects(std::declval<A>()...));
  template <class... A>
  using NearestCall = decltype(std::declval<const MeshGrid&>().nearestObject(std::declval<A>()...));
  template <class... A>
  using NearestNCall = decltype(std::declval<const MeshGrid&>().nearestObjects(std::declval<A>()...));

  template <class, template <class...> class, class...>
  struct Detect : std::false_type { using type = void; };
  template <template <class...> class Op, class... A>
  struct Detect<std::void_t<Op<A...>>, Op, A...> : std::true_type { using type = Op<A...>; };

  template <class R, template <class...> class Op, class... A>
  constexpr bool resolvesTo = Detect<void, Op, A...>::value && std::is_same<typename Detect<void, Op, A...>::type, R>::value;
  template <template <class...> class Op, class... A>
  constexpr bool compiles = Detect<void, Op, A...>::value;

  using Doubles = std::vector<double>;
  using Batch = MeshGrid::Batch;

  static_assert(resolvesTo<Batch, CloseCall, Indices&, double, double>, "index batch, non-const vector");
  static_assert(resolvesTo<Batch, CloseCall, Indices&, int, double>, "index batch, integer Rmax");
  static_assert(resolvesTo<Batch, CloseCall, Indices&, int, int>, "index batch, integer radii");
  static_assert(resolvesTo<Batch, CloseCall, Indices&, double>, "index batch, no Rmin");
  static_assert(resolvesTo<Batch, CloseCall, const Indices&, double, double>, "index batch, const vector");
  static_assert(resolvesTo<void, CloseCall, Indices&, double, Batch&>, "index batch into Batch, no Rmin");
  static_assert(resolvesTo<void, CloseCall, Indices&, double, double, Batch&>, "index batch into Batch");
  static_assert(resolvesTo<Batch, CloseCall, Doubles&, Doubles&, Doubles&, double>, "point batch");
  static_assert(resolvesTo<void, CloseCall, Doubles&, Doubles&, Doubles&, double, Batch&>, "point batch into Batch, no Rmin");
  static_assert(resolvesTo<void, CloseCall, Doubles&, Doubles&, Doubles&, double, double, Batch&>, "point batch into Batch");
  static_assert(resolvesTo<Indices, CloseCall, unsigned, double>, "index query");
  static_assert(resolvesTo<Indices, CloseCall, int, int, int>, "index query, integers");
  static_assert(resolvesTo<void, CloseCall, unsigned, double, Indices&>, "index query into a vector, no Rmin");
  static_assert(resolvesTo<void, CloseCall, int, double, double, Indices&>, "index query into a vector");
  static_assert(resolvesTo<Indices, CloseCall, double, double, double, double>, "point query");
  static_assert(resolvesTo<Indices, CloseCall, int, int, int, int>, "point query, integers");
  static_assert(resolvesTo<Indices, CloseCall, int, int, int, int, int>, "point query with Rmin, integers");
  static_assert(resolvesTo<void, CloseCall, double, double, double, double, Indices&>, "point query into a vector, no Rmin");
  static_assert(resolvesTo<void, CloseCall, double, double, double, double, double, Indices&>, "point query into a vector");

  static_assert(!compiles<CloseCall, unsigned, double, const Indices&>, "a const container is not an output");
  static_assert(!compiles<CloseCall, unsigned, double, Counts&>, "closeObjects does not fill counts");
  static_assert(!compiles<CloseCall, unsigned, double, Batch&>, "a single query does not fill a Batch");
  static_assert(!compiles<CloseCall, double, double, double, double, Batch&>, "a single point query does not fill a Batch");
  static_assert(!compiles<CloseCall, Indices&, double, Indices&>, "a batch fills a Batch, not a vector");
  static_assert(!compiles<CloseCall, Doubles&, Doubles&, Doubles&, double, Indices&>, "a point batch fills a Batch");
  static_assert(!compiles<CloseCall, Indices&, Indices&>, "two vectors are no call");

  static_assert(resolvesTo<Counts, CountCall, Indices&, int, int>, "count index batch");
  static_assert(resolvesTo<Counts, CountCall, Indices&, double>, "count index batch, no Rmin");
  static_assert(resolvesTo<void, CountCall, Indices&, double, Counts&>, "count index batch into counts");
  static_assert(resolvesTo<void, CountCall, Indices&, double, double, Counts&>, "count index batch into counts, Rmin");
  static_assert(resolvesTo<Counts, CountCall, Doubles&, Doubles&, Doubles&, double>, "count point batch");
  static_assert(resolvesTo<void, CountCall, Doubles&, Doubles&, Doubles&, double, Counts&>, "count point batch into counts");
  static_assert(resolvesTo<std::size_t, CountCall, unsigned, double>, "count index query");
  static_assert(resolvesTo<std::size_t, CountCall, double, double, double, double>, "count point query");
  static_assert(!compiles<CountCall, Indices&, double, Indices&>, "counts are 64-bit");
  static_assert(!compiles<CountCall, unsigned, double, Counts&>, "a single count returns its value");

  static_assert(resolvesTo<unsigned int, NearestCall, unsigned>, "nearest by index");
  static_assert(resolvesTo<unsigned int, NearestCall, int, int, int>, "nearest to a point, integers");
  static_assert(resolvesTo<Indices, NearestCall, Indices&>, "nearest index batch");
  static_assert(resolvesTo<void, NearestCall, Indices&, Indices&>, "nearest index batch into a vector");
  static_assert(resolvesTo<Indices, NearestCall, Doubles&, Doubles&, Doubles&>, "nearest point batch");
  static_assert(resolvesTo<void, NearestCall, Doubles&, Doubles&, Doubles&, Indices&>, "nearest point batch into a vector");
  static_assert(!compiles<NearestCall, unsigned, Indices&>, "a single nearestObject returns its value");
  static_assert(!compiles<NearestCall, Indices&, Counts&>, "nearestObject fills indices");

  static_assert(resolvesTo<Indices, NearestNCall, unsigned, unsigned>, "N nearest by index");
  static_assert(resolvesTo<Indices, NearestNCall, int, int>, "N nearest by index, integers");
  static_assert(resolvesTo<Indices, NearestNCall, int, int, int, int>, "N nearest to a point, integers");
  static_assert(resolvesTo<void, NearestNCall, unsigned, unsigned, Indices&>, "N nearest by index into a vector");
  static_assert(resolvesTo<void, NearestNCall, unsigned, double, double, double, Indices&>, "N nearest to a point into a vector");
  static_assert(resolvesTo<Indices, NearestNCall, unsigned, Indices&>, "N nearest index batch");
  static_assert(resolvesTo<void, NearestNCall, unsigned, Indices&, Indices&>, "N nearest index batch into a vector");
  static_assert(resolvesTo<Indices, NearestNCall, unsigned, Doubles&, Doubles&, Doubles&>, "N nearest point batch");
  static_assert(resolvesTo<void, NearestNCall, unsigned, Doubles&, Doubles&, Doubles&, Indices&>, "N nearest point batch into a vector");
  static_assert(!compiles<NearestNCall, Indices&, unsigned, unsigned>, "the container no longer comes first");
  static_assert(!compiles<NearestNCall, unsigned, unsigned, Counts&>, "nearestObjects fills indices");

  // Workers is the last argument of every batch form, after the defaulted
  // Rmin, and of no single query. It is no number: Workers{8} cannot take
  // Rmin's place, and no integer becomes a Workers without being named.
  using meshsearch::Workers;
  static_assert(!std::is_convertible<int, Workers>::value && !std::is_convertible<Workers, int>::value &&
                !std::is_convertible<Workers, double>::value && std::is_constructible<Workers, int>::value,
                "Workers converts to and from no number");
  static_assert(resolvesTo<Indices, NearestCall, Doubles&, Doubles&, Doubles&, Workers>, "nearest point batch, workers");
  static_assert(resolvesTo<void, NearestCall, Doubles&, Doubles&, Doubles&, Indices&, Workers>, "into a vector, workers");
  static_assert(resolvesTo<Indices, NearestCall, Indices&, Workers>, "nearest index batch, workers");
  static_assert(resolvesTo<void, NearestCall, Indices&, Indices&, Workers>, "nearest index batch into a vector, workers");
  static_assert(resolvesTo<Indices, NearestNCall, unsigned, Doubles&, Doubles&, Doubles&, Workers>, "N nearest point batch, workers");
  static_assert(resolvesTo<void, NearestNCall, unsigned, Doubles&, Doubles&, Doubles&, Indices&, Workers>, "into a vector, workers");
  static_assert(resolvesTo<Indices, NearestNCall, unsigned, Indices&, Workers>, "N nearest index batch, workers");
  static_assert(resolvesTo<void, NearestNCall, unsigned, Indices&, Indices&, Workers>, "into a vector, workers");
  static_assert(resolvesTo<Batch, CloseCall, Doubles&, Doubles&, Doubles&, double, double, Workers>, "point batch, workers");
  static_assert(resolvesTo<void, CloseCall, Doubles&, Doubles&, Doubles&, double, double, Batch&, Workers>, "into Batch, workers");
  static_assert(resolvesTo<void, CloseCall, Doubles&, Doubles&, Doubles&, double, Batch&, Workers>, "no Rmin, workers");
  static_assert(resolvesTo<Batch, CloseCall, Indices&, double, double, Workers>, "index batch, workers");
  static_assert(resolvesTo<void, CloseCall, Indices&, double, double, Batch&, Workers>, "index batch into Batch, workers");
  static_assert(resolvesTo<void, CloseCall, Indices&, double, Batch&, Workers>, "index batch, no Rmin, workers");
  static_assert(resolvesTo<Counts, CountCall, Doubles&, Doubles&, Doubles&, double, double, Workers>, "count point batch, workers");
  static_assert(resolvesTo<void, CountCall, Doubles&, Doubles&, Doubles&, double, double, Counts&, Workers>, "into counts, workers");
  static_assert(resolvesTo<void, CountCall, Doubles&, Doubles&, Doubles&, double, Counts&, Workers>, "no Rmin, workers");
  static_assert(resolvesTo<Counts, CountCall, Indices&, double, double, Workers>, "count index batch, workers");
  static_assert(resolvesTo<void, CountCall, Indices&, double, double, Counts&, Workers>, "count index into counts, workers");
  static_assert(resolvesTo<void, CountCall, Indices&, double, Counts&, Workers>, "count index, no Rmin, workers");
  static_assert(!compiles<CloseCall, Indices&, double, Workers>, "Workers cannot stand for Rmin");
  static_assert(!compiles<CloseCall, Doubles&, Doubles&, Doubles&, double, Workers>, "nor in the point batch");
  static_assert(!compiles<CountCall, Indices&, double, Workers>, "nor in count");
  static_assert(!compiles<CloseCall, Indices&, double, double, int>, "a number is no Workers");
  static_assert(!compiles<CountCall, Indices&, double, double, Counts&, int>, "nor after the container");
  static_assert(!compiles<NearestCall, Indices&, int>, "nor in nearestObject");
  static_assert(!compiles<CloseCall, unsigned, double, double, Workers>, "a single query takes no Workers");
  static_assert(!compiles<CountCall, double, double, double, double, double, Workers>, "nor a single count");
  static_assert(!compiles<NearestCall, unsigned, Workers>, "nor a single nearestObject");
  static_assert(!compiles<NearestNCall, unsigned, unsigned, Indices&, Workers>, "nor a single nearestObjects");

  template <class F>
  bool raises (F&& f)
  {
    try { f(); }
    catch (const IndexError&) { return false; }
    catch (const Error&) { return true; }
    return false;
  }

  template <class F>
  bool raisesIndex (F&& f)
  {
    try { f(); }
    catch (const IndexError&) { return true; }
    catch (const Error&) { return false; }
    return false;
  }

  // A container holding stale entries, to check that the forms clear it.
  Indices stale () { return Indices(7, 12345u); }

  struct Case {
    const char* name;
    MeshGrid grid;
    std::vector<double> qx, qy, qz;   // query points, all inside the box
    Indices qi;                       // query indices, all alive
    std::vector<double> radii;        // Rmax values; Rmin is drawn below them
  };

  // Query points: random ones, plus the corners and face centres of the box
  // and points on cell faces, where the cell a query starts from is decided by
  // rounding.
  void addQueries (Case& c, const int nRandom)
  {
    const auto box = c.grid.get_lims();
    const double cs = c.grid.get_cellsize();
    for (int i=0; i<nRandom; ++i) {
      c.qx.push_back(uniform(box[0][0], box[0][1]));
      c.qy.push_back(uniform(box[1][0], box[1][1]));
      c.qz.push_back(uniform(box[2][0], box[2][1]));
    }
    for (int corner=0; corner<8; ++corner) {
      c.qx.push_back(box[0][corner & 1]);
      c.qy.push_back(box[1][(corner >> 1) & 1]);
      c.qz.push_back(box[2][(corner >> 2) & 1]);
    }
    for (int i=0; i<8; ++i) {
      const auto n = c.grid.get_nCells();
      c.qx.push_back(std::min(box[0][1], box[0][0]+cs*integer(0, int(n[0]))));
      c.qy.push_back(std::min(box[1][1], box[1][0]+cs*integer(0, int(n[1]))));
      c.qz.push_back(std::min(box[2][1], box[2][0]+cs*integer(0, int(n[2]))));
    }
    for (unsigned int i=0; i<c.grid.get_nObjects()+50 && c.qi.size()<200; ++i)
      if (c.grid.isAlive(i)) c.qi.push_back(i);
    std::shuffle(c.qi.begin(), c.qi.end(), rng);
  }

  // ------------------------------------------------------------------
  // Single-query forms: container-first equals returning; countObjects equals
  // the size of closeObjects.
  // ------------------------------------------------------------------

  void checkSingle (const Case& c)
  {
    const MeshGrid& g = c.grid;
    const unsigned int n = g.get_nObjects();
    for (size_t q=0; q<c.qx.size(); ++q) {
      const double x = c.qx[q], y = c.qy[q], z = c.qz[q];
      for (const double R : c.radii) {
        const double Rmin = (q % 3 == 0) ? 0. : R*uniform(0., 1.);
        Indices out = stale();
        g.closeObjects(x, y, z, R, Rmin, out);
        assert(out == g.closeObjects(x, y, z, R, Rmin));
        Indices ball = stale();
        g.closeObjects(x, y, z, R, ball);
        assert(ball == g.closeObjects(x, y, z, R));
        assert(g.countObjects(x, y, z, R, Rmin) == out.size());
      }
      for (const unsigned int N : {0u, 1u, 3u, std::min(n, 60u), std::min(n, 90u), n}) {
        Indices out = stale();
        g.nearestObjects(N, x, y, z, out);
        assert(out == g.nearestObjects(N, x, y, z));
      }
    }
    for (const unsigned int i : c.qi) {
      for (const double R : c.radii) {
        const double Rmin = (i % 3 == 0) ? 0. : R*uniform(0., 1.);
        Indices out = stale();
        g.closeObjects(i, R, Rmin, out);
        assert(out == g.closeObjects(i, R, Rmin));
        Indices ball = stale();
        g.closeObjects(i, R, ball);
        assert(ball == g.closeObjects(i, R));
        assert(g.countObjects(i, R, Rmin) == out.size());
      }
      for (const unsigned int N : {0u, 1u, 3u, std::min(n-1, 60u), std::min(n-1, 90u), n-1}) {
        Indices out = stale();
        g.nearestObjects(N, i, out);
        assert(out == g.nearestObjects(N, i));
      }
    }
  }

  // ------------------------------------------------------------------
  // Batch forms: each query's result, in input order, equals the single query.
  // ------------------------------------------------------------------

  void checkBatch (const Case& c)
  {
    const MeshGrid& g = c.grid;
    const unsigned int n = g.get_nObjects();
    const size_t nq = c.qx.size();

    {
      Indices expect;
      for (size_t q=0; q<nq; ++q) expect.push_back(g.nearestObject(c.qx[q], c.qy[q], c.qz[q]));
      assert(g.nearestObject(c.qx, c.qy, c.qz) == expect);
      Indices out = stale();
      g.nearestObject(c.qx, c.qy, c.qz, out);
      assert(out == expect);
    }
    if (n >= 2) {
      Indices expect;
      for (const unsigned int i : c.qi) expect.push_back(g.nearestObject(i));
      assert(g.nearestObject(c.qi) == expect);
      Indices out = stale();
      g.nearestObject(c.qi, out);
      assert(out == expect);
      Indices same = c.qi;
      g.nearestObject(same, same);
      assert(same == expect);
    }

    for (const unsigned int N : {0u, 1u, 3u, std::min(n, 60u), std::min(n, 90u), n}) {
      Indices expect;
      for (size_t q=0; q<nq; ++q) {
        const Indices one = g.nearestObjects(N, c.qx[q], c.qy[q], c.qz[q]);
        assert(one.size() == N);
        expect.insert(expect.end(), one.begin(), one.end());
      }
      assert(g.nearestObjects(N, c.qx, c.qy, c.qz) == expect);
      Indices out = stale();
      g.nearestObjects(N, c.qx, c.qy, c.qz, out);
      assert(out == expect);
    }
    if (n >= 1) {
      for (const unsigned int N : {0u, 1u, 3u, std::min(n-1, 60u), std::min(n-1, 90u), n-1}) {
        Indices expect;
        for (const unsigned int i : c.qi) {
          const Indices one = g.nearestObjects(N, i);
          expect.insert(expect.end(), one.begin(), one.end());
        }
        assert(g.nearestObjects(N, c.qi) == expect);
        Indices out = stale();
        g.nearestObjects(N, c.qi, out);
        assert(out == expect);
        Indices same = c.qi;
        g.nearestObjects(N, same, same);
        assert(same == expect);
      }
    }

    for (const double R : c.radii) {
      for (const double Rmin : {0., 0.5*R, R}) {
        MeshGrid::Batch expect;
        expect.offsets.push_back(0);
        Counts counts;
        for (size_t q=0; q<nq; ++q) {
          const Indices one = g.closeObjects(c.qx[q], c.qy[q], c.qz[q], R, Rmin);
          expect.indices.insert(expect.indices.end(), one.begin(), one.end());
          expect.offsets.push_back(expect.indices.size());
          counts.push_back(one.size());
        }
        const MeshGrid::Batch got = g.closeObjects(c.qx, c.qy, c.qz, R, Rmin);
        assert(got.indices == expect.indices && got.offsets == expect.offsets);
        MeshGrid::Batch out{stale(), Counts(3, 99)};
        g.closeObjects(c.qx, c.qy, c.qz, R, Rmin, out);
        assert(out.indices == expect.indices && out.offsets == expect.offsets);
        if (Rmin == 0.) {
          MeshGrid::Batch ball{stale(), Counts(3, 99)};
          g.closeObjects(c.qx, c.qy, c.qz, R, ball);
          assert(ball.indices == expect.indices && ball.offsets == expect.offsets);
          Counts cball(5, 99);
          g.countObjects(c.qx, c.qy, c.qz, R, cball);
          assert(cball == counts);
        }
        assert(g.countObjects(c.qx, c.qy, c.qz, R, Rmin) == counts);
        Counts cout(5, 99);
        g.countObjects(c.qx, c.qy, c.qz, R, Rmin, cout);
        assert(cout == counts);
      }
      for (const double Rmin : {0., 0.5*R}) {
        MeshGrid::Batch expect;
        expect.offsets.push_back(0);
        Counts counts;
        for (const unsigned int i : c.qi) {
          const Indices one = g.closeObjects(i, R, Rmin);
          expect.indices.insert(expect.indices.end(), one.begin(), one.end());
          expect.offsets.push_back(expect.indices.size());
          counts.push_back(one.size());
        }
        const MeshGrid::Batch got = g.closeObjects(c.qi, R, Rmin);
        assert(got.indices == expect.indices && got.offsets == expect.offsets);
        MeshGrid::Batch out{stale(), Counts(3, 99)};
        g.closeObjects(c.qi, R, Rmin, out);
        assert(out.indices == expect.indices && out.offsets == expect.offsets);
        // An index batch answered into the Batch that holds its indices: the
        // header allows it, so the answer must be the separate one's.
        MeshGrid::Batch same{c.qi, Counts(3, 99)};
        g.closeObjects(same.indices, R, Rmin, same);
        assert(same.indices == expect.indices && same.offsets == expect.offsets);
        if (Rmin == 0.) {
          MeshGrid::Batch ball{stale(), Counts(3, 99)};
          g.closeObjects(c.qi, R, ball);
          assert(ball.indices == expect.indices && ball.offsets == expect.offsets);
          Counts cball(5, 99);
          g.countObjects(c.qi, R, cball);
          assert(cball == counts);
        }
        assert(g.countObjects(c.qi, R, Rmin) == counts);
        Counts cout(5, 99);
        g.countObjects(c.qi, R, Rmin, cout);
        assert(cout == counts);
      }
    }
  }

  // ------------------------------------------------------------------
  // Errors: a batch raises if any query would; shared arguments are checked
  // even with no query; an empty batch returns empty results.
  // ------------------------------------------------------------------

  void checkErrors (const Case& c)
  {
    const MeshGrid& g = c.grid;
    const unsigned int n = g.get_nObjects();
    const auto box = g.get_lims();

    std::vector<double> bx = c.qx, by = c.qy, bz = c.qz;
    bx[bx.size()/2] = box[0][1]+(box[0][1]-box[0][0]);
    Indices out = stale();
    Counts cout;
    MeshGrid::Batch bout;
    assert(raises([&] { g.nearestObject(bx, by, bz); }));
    assert(raises([&] { g.nearestObjects(1, bx, by, bz); }));
    assert(raises([&] { g.nearestObjects(0, bx, by, bz); }));
    assert(raises([&] { g.closeObjects(bx, by, bz, 1.); }));
    assert(raises([&] { g.countObjects(bx, by, bz, 1.); }));
    assert(raises([&] { g.nearestObject(bx, by, bz, out); }));
    assert(raises([&] { g.closeObjects(bx, by, bz, 1., bout); }));
    assert(raises([&] { g.countObjects(bx, by, bz, 1., cout); }));
    bz.pop_back();
    assert(raises([&] { g.nearestObject(c.qx, c.qy, bz); }));
    assert(raises([&] { g.nearestObjects(1, c.qx, c.qy, bz); }));
    assert(raises([&] { g.closeObjects(c.qx, c.qy, bz, 1.); }));
    assert(raises([&] { g.countObjects(c.qx, c.qy, bz, 1.); }));

    for (const double bad : {-1., std::nan("")}) {
      assert(raises([&] { g.countObjects(c.qx[0], c.qy[0], c.qz[0], bad); }));
      assert(raises([&] { g.closeObjects(c.qx, c.qy, c.qz, bad); }));
      assert(raises([&] { g.countObjects(c.qx, c.qy, c.qz, bad); }));
      assert(raises([&] { g.closeObjects(std::vector<double>{}, {}, {}, bad); }));
      assert(raises([&] { g.countObjects(Indices{}, bad); }));
      assert(raises([&] { g.closeObjects(c.qx[0], c.qy[0], c.qz[0], bad, out); }));
    }
    assert(raises([&] { g.countObjects(c.qx[0], c.qy[0], c.qz[0], 1., 2.); }));
    assert(raises([&] { g.countObjects(box[0][1]+1., c.qy[0], c.qz[0], 1.); }));
    assert(raises([&] { g.closeObjects(box[0][1]+1., c.qy[0], c.qz[0], 1., out); }));
    assert(raises([&] { g.nearestObjects(1, box[0][1]+1., c.qy[0], c.qz[0], out); }));

    assert(raises([&] { g.nearestObjects(n+1, std::vector<double>{}, {}, {}); }));
    assert(raises([&] { g.nearestObjects(n, Indices{}); }));
    assert(raises([&] { g.nearestObjects(n+1, c.qx[0], c.qy[0], c.qz[0], out); }));

    const unsigned int beyond = g.get_nObjects()+1000;
    Indices bi = c.qi;
    bi.insert(bi.begin()+bi.size()/2, beyond);
    assert(raisesIndex([&] { g.nearestObject(bi); }));
    assert(raisesIndex([&] { g.nearestObjects(1, bi); }));
    assert(raisesIndex([&] { g.nearestObjects(0, bi); }));
    assert(raisesIndex([&] { g.closeObjects(bi, 1.); }));
    assert(raisesIndex([&] { g.countObjects(bi, 1.); }));
    assert(raisesIndex([&] { g.countObjects(beyond, 1.); }));
    assert(raisesIndex([&] { g.closeObjects(beyond, 1., out); }));
    assert(raisesIndex([&] { g.nearestObjects(1, beyond, out); }));

    const std::vector<double> none;
    assert(g.nearestObject(none, none, none).empty());
    assert(g.nearestObject(Indices{}).empty());
    assert(g.nearestObjects(3, none, none, none).empty());
    assert(g.nearestObjects(3, Indices{}).empty());
    const MeshGrid::Batch empty = g.closeObjects(none, none, none, 1.);
    assert(empty.indices.empty() && empty.offsets == Counts{0});
    MeshGrid::Batch emptyOut{stale(), Counts(3, 99)};
    g.closeObjects(Indices{}, 1., emptyOut);
    assert(emptyOut.indices.empty() && emptyOut.offsets == Counts{0});
    assert(g.countObjects(none, none, none, 1.).empty());
  }

  void checkCase (const Case& c)
  {
    checkSingle(c);
    checkBatch(c);
    checkErrors(c);
    std::printf("  %-34s %6u objects, %4zu points, %4zu indices: ok\n",
                c.name, c.grid.get_nObjects(), c.qx.size(), c.qi.size());
  }

  void randomPoints (std::vector<double>& x, std::vector<double>& y, std::vector<double>& z,
                     const int n, const Box& box)
  {
    for (int i=0; i<n; ++i) {
      x.push_back(uniform(box[0][0], box[0][1]));
      y.push_back(uniform(box[1][0], box[1][1]));
      z.push_back(uniform(box[2][0], box[2][1]));
    }
  }

  // A large batch, point and index forms, against the single queries
  // concatenated in input order: same indices, same order, same offsets, and
  // the same count for each query. Then with 3 workers and with every available
  // core (-1), which must change nothing; the other counts, and smaller batches,
  // are in test_workers.
  void checkLarge (const MeshGrid& g, const std::vector<double>& qx, const std::vector<double>& qy,
                   const std::vector<double>& qz, const Indices& qi, const double Rmax, const double Rmin)
  {
    MeshGrid::Batch expect, expecti;
    expect.offsets.push_back(0);
    expecti.offsets.push_back(0);
    Counts counts, countsi;
    Indices one;
    for (size_t q=0; q<qx.size(); ++q) {
      g.closeObjects(qx[q], qy[q], qz[q], Rmax, Rmin, one);
      expect.indices.insert(expect.indices.end(), one.begin(), one.end());
      expect.offsets.push_back(expect.indices.size());
      counts.push_back(g.countObjects(qx[q], qy[q], qz[q], Rmax, Rmin));
    }
    for (const unsigned int i : qi) {
      g.closeObjects(i, Rmax, Rmin, one);
      expecti.indices.insert(expecti.indices.end(), one.begin(), one.end());
      expecti.offsets.push_back(expecti.indices.size());
      countsi.push_back(g.countObjects(i, Rmax, Rmin));
      assert(countsi.back() == one.size());
    }
    assert(g.countObjects(qx, qy, qz, Rmax, Rmin) == counts);
    assert(g.countObjects(qi, Rmax, Rmin) == countsi);
    Counts countOut(5, 7);
    g.countObjects(qi, Rmax, Rmin, countOut);
    assert(countOut == countsi);
    if (Rmin == 0.) assert(g.countObjects(qi, Rmax) == countsi);
    const MeshGrid::Batch b = g.closeObjects(qx, qy, qz, Rmax, Rmin);
    assert(b.indices == expect.indices && b.offsets == expect.offsets);
    const MeshGrid::Batch bi = g.closeObjects(qi, Rmax, Rmin);
    assert(bi.indices == expecti.indices && bi.offsets == expecti.offsets);
    MeshGrid::Batch out{Indices(7, 5), Counts(3, 99)};
    g.closeObjects(qi, Rmax, Rmin, out);
    assert(out.indices == expecti.indices && out.offsets == expecti.offsets);
    // Same, aliased, on the path that answers in cell order.
    MeshGrid::Batch same{qi, Counts(3, 99)};
    g.closeObjects(same.indices, Rmax, Rmin, same);
    assert(same.indices == expecti.indices && same.offsets == expecti.offsets);

    for (const int w : {3, -1}) {
      const meshsearch::Workers W(w);
      assert(g.countObjects(qx, qy, qz, Rmax, Rmin, W) == counts);
      countOut.assign(5, 7);
      g.countObjects(qi, Rmax, Rmin, countOut, W);
      assert(countOut == countsi);
      const MeshGrid::Batch bw = g.closeObjects(qx, qy, qz, Rmax, Rmin, W);
      assert(bw.indices == expect.indices && bw.offsets == expect.offsets);
      MeshGrid::Batch outw{Indices(7, 5), Counts(3, 99)};
      g.closeObjects(qi, Rmax, Rmin, outw, W);
      assert(outw.indices == expecti.indices && outw.offsets == expecti.offsets);
    }
  }

}


int main ()
{
  std::printf("container-first, count and batch forms against the single queries\n");

  const Box unit = {{{0., 1.}, {0., 1.}, {0., 1.}}};

  // Random data, dense (several objects per cell) and sparse (most cells
  // empty, so the walks reach far shells).
  for (const auto& regime : {std::array<double,2>{2000., 0.05}, std::array<double,2>{60., 0.08}}) {
    std::vector<double> x, y, z;
    randomPoints(x, y, z, int(regime[0]), unit);
    Case c{regime[0] > 100. ? "random dense" : "random sparse", MeshGrid(x, y, z, regime[1]), {}, {}, {}, {}, {}};
    c.radii = {0., 0.03, 0.1, 0.4, inf};
    addQueries(c, 150);
    checkCase(c);
  }

  // Periodic: all axes, then one axis on a long box. Queries near the faces
  // see neighbours through the wrap; the half-period points sit where the
  // image choice is made.
  for (const auto& per : {std::array<bool,3>{true, true, true}, std::array<bool,3>{false, true, false}}) {
    MeshGrid::Options opt;
    opt.limits = Box{{{-2., 2.}, {0., 8.}, {10., 14.}}};
    opt.periodic = per;
    std::vector<double> x, y, z;
    randomPoints(x, y, z, 1500, *opt.limits);
    for (int i=0; i<40; ++i) {
      x.push_back((i & 1) ? -2. : 2.);
      y.push_back(4.*(i % 3));
      z.push_back(12.+0.5*(i % 5)-1.);
    }
    Case c{per[0] ? "periodic xyz" : "periodic y", MeshGrid(x, y, z, 0.37, opt), {}, {}, {}, {}, {}};
    c.radii = {0., 0.2, 0.9, 2.5, 7., inf};
    addQueries(c, 150);
    c.qx.push_back(0.); c.qy.push_back(4.); c.qz.push_back(12.);
    c.qx.push_back(-2.); c.qy.push_back(0.); c.qz.push_back(10.);
    checkCase(c);
  }

  // Adversarial: an integer lattice, so that distances tie in large groups and
  // nearestObjects must break them by index at the last place, with radii
  // equal to lattice distances so that closed-interval ends are hit exactly;
  // coincident objects; removed and added objects, so that live indices have
  // gaps and cells hold moved blocks.
  {
    std::vector<double> x, y, z;
    for (int i=0; i<9; ++i) for (int j=0; j<9; ++j) for (int k=0; k<9; ++k) {
      x.push_back(i); y.push_back(j); z.push_back(k);
    }
    for (int i=0; i<30; ++i) { x.push_back(4.); y.push_back(4.); z.push_back(4.); }
    MeshGrid::Options opt;
    opt.limits = Box{{{0., 8.}, {0., 8.}, {0., 8.}}};
    Case c{"lattice, ties, coincident, edits", MeshGrid(x, y, z, 1., opt), {}, {}, {}, {}, {}};
    for (unsigned int i=0; i<x.size(); i+=7) c.grid.removeObject(i);
    for (int i=0; i<200; ++i) c.grid.addObject(integer(0, 8), integer(0, 8), integer(0, 8));
    c.radii = {0., 1., std::sqrt(2.), std::sqrt(3.), 2., 3., inf};
    addQueries(c, 60);
    for (int i=0; i<40; ++i) {
      c.qx.push_back(integer(0, 16)*0.5);
      c.qy.push_back(integer(0, 16)*0.5);
      c.qz.push_back(integer(0, 16)*0.5);
    }
    checkCase(c);
  }

  {
    MeshGrid::Options opt;
    opt.limits = Box{{{0., 6.}, {0., 6.}, {0., 6.}}};
    opt.periodic = {true, true, true};
    std::vector<double> x, y, z;
    for (int i=0; i<6; ++i) for (int j=0; j<6; ++j) for (int k=0; k<6; ++k) {
      x.push_back(i); y.push_back(j); z.push_back(k);
    }
    Case c{"periodic lattice, half-period ties", MeshGrid(x, y, z, 1., opt), {}, {}, {}, {}, {}};
    c.radii = {0., 1., 2., 3., std::sqrt(27.), inf};
    addQueries(c, 40);
    for (int i=0; i<40; ++i) {
      c.qx.push_back(integer(0, 12)*0.5);
      c.qy.push_back(integer(0, 12)*0.5);
      c.qz.push_back(integer(0, 12)*0.5);
    }
    checkCase(c);
  }

  // Large batches. A batch may answer its queries in an order of its own
  // choosing (by cell, for locality) once it holds enough of them; results
  // must still come back in input order. 120000 queries is above the size at
  // which the library switches (100000), so this reaches that path; the
  // checks hold whatever the switch point is. Indices repeat and come in
  // random order.
  for (const bool periodic : {false, true}) {
    MeshGrid::Options opt;
    opt.limits = unit;
    opt.periodic = {periodic, periodic, periodic};
    std::vector<double> x, y, z;
    randomPoints(x, y, z, 30000, unit);
    Case c{periodic ? "large batch, periodic" : "large batch", MeshGrid(x, y, z, 0.03, opt), {}, {}, {}, {}, {}};
    randomPoints(c.qx, c.qy, c.qz, 120000, unit);
    for (int i=0; i<120000; ++i) c.qi.push_back(unsigned(integer(0, 29999)));
    const MeshGrid& g = c.grid;
    const size_t nq = c.qx.size();

    Indices nn, knn, nni, knni;
    MeshGrid::Batch close, closei;
    close.offsets.push_back(0);
    closei.offsets.push_back(0);
    Counts counts, countsi;
    for (size_t q=0; q<nq; ++q) {
      nn.push_back(g.nearestObject(c.qx[q], c.qy[q], c.qz[q]));
      const Indices k = g.nearestObjects(10, c.qx[q], c.qy[q], c.qz[q]);
      knn.insert(knn.end(), k.begin(), k.end());
      const Indices one = g.closeObjects(c.qx[q], c.qy[q], c.qz[q], 0.05, 0.01);
      close.indices.insert(close.indices.end(), one.begin(), one.end());
      close.offsets.push_back(close.indices.size());
      counts.push_back(one.size());
    }
    for (const unsigned int i : c.qi) {
      nni.push_back(g.nearestObject(i));
      const Indices k = g.nearestObjects(70, i);
      knni.insert(knni.end(), k.begin(), k.end());
      const Indices one = g.closeObjects(i, 0.05);
      closei.indices.insert(closei.indices.end(), one.begin(), one.end());
      closei.offsets.push_back(closei.indices.size());
      countsi.push_back(one.size());
    }
    assert(g.nearestObject(c.qx, c.qy, c.qz) == nn);
    assert(g.nearestObjects(10, c.qx, c.qy, c.qz) == knn);
    const MeshGrid::Batch b = g.closeObjects(c.qx, c.qy, c.qz, 0.05, 0.01);
    assert(b.indices == close.indices && b.offsets == close.offsets);
    assert(g.countObjects(c.qx, c.qy, c.qz, 0.05, 0.01) == counts);
    assert(g.nearestObject(c.qi) == nni);
    assert(g.nearestObjects(70, c.qi) == knni);
    const MeshGrid::Batch bi = g.closeObjects(c.qi, 0.05);
    assert(bi.indices == closei.indices && bi.offsets == closei.offsets);
    const MeshGrid::Batch bi0 = g.closeObjects(c.qi, 0.05, 0.);
    assert(bi0.indices == closei.indices && bi0.offsets == closei.offsets);
    assert(g.countObjects(c.qi, 0.05) == countsi);
    // The same batches with several workers, which must change nothing.
    for (const int w : {3, -1}) {
      const meshsearch::Workers W(w);
      assert(g.nearestObject(c.qx, c.qy, c.qz, W) == nn);
      assert(g.nearestObjects(10, c.qx, c.qy, c.qz, W) == knn);
      assert(g.nearestObject(c.qi, W) == nni);
      assert(g.nearestObjects(70, c.qi, W) == knni);
      const MeshGrid::Batch bw = g.closeObjects(c.qx, c.qy, c.qz, 0.05, 0.01, W);
      assert(bw.indices == close.indices && bw.offsets == close.offsets);
      assert(g.countObjects(c.qx, c.qy, c.qz, 0.05, 0.01, W) == counts);
      const MeshGrid::Batch biw = g.closeObjects(c.qi, 0.05, 0., W);
      assert(biw.indices == closei.indices && biw.offsets == closei.offsets);
      assert(g.countObjects(c.qi, 0.05, 0., W) == countsi);
    }

    std::vector<double> bx = c.qx;
    bx[nq-3] = 2.;
    assert(raises([&] { g.nearestObjects(10, bx, c.qy, c.qz); }));
    Indices bad = c.qi;
    bad[nq/3] = 40000;
    assert(raisesIndex([&] { g.closeObjects(bad, 0.05); }));
    std::printf("  %-34s %6u objects, %5zu points, %5zu indices: ok\n",
                c.name, g.get_nObjects(), nq, c.qi.size());
  }

  // Large batches on every path a query can take. Each configuration below
  // holds at least 100000 queries, the size from which the library answers
  // a batch in an order of its own, and is checked in both forms.
  //
  // Sparse grid (whole-cell rules off), plain and periodic: both radius
  // forms, a ball and a shell.
  for (const bool periodic : {false, true}) {
    MeshGrid::Options opt;
    opt.limits = unit;
    opt.periodic = {periodic, periodic, periodic};
    std::vector<double> x, y, z, qx, qy, qz;
    randomPoints(x, y, z, 30000, unit);
    randomPoints(qx, qy, qz, 100000, unit);
    Indices qi;
    for (int i=0; i<100000; ++i) qi.push_back(unsigned(integer(0, 29999)));
    const MeshGrid g(x, y, z, 0.03, opt);
    checkLarge(g, qx, qy, qz, qi, 0.05, 0.);
    checkLarge(g, qx, qy, qz, qi, 0.06, 0.02);
  }
  std::printf("  large batches, sparse grid:        ok\n");

  // Dense grid at Rmax >= 12 cell sides, where whole cells are accepted or
  // skipped without measuring each object. Cells are 1/8 of the unit box
  // (an integer number, as a periodic axis requires), so the walk stays
  // short. Objects come in 60 clumps of 5 inside single cells, so the grid
  // is dense (at least 4 objects per occupied cell). From a query near a
  // corner of the plain box, a ball of 12 cells leaves the far corner out,
  // so clumps fall inside, across the sphere and outside; Rmin = 0.3 adds
  // cells skipped wholly inside the inner sphere, on the periodic grid too.
  for (const bool periodic : {false, true}) {
    MeshGrid::Options opt;
    opt.limits = unit;
    opt.periodic = {periodic, periodic, periodic};
    const double c = 1./8.;
    std::vector<double> x, y, z, qx, qy, qz;
    for (int k=0; k<60; ++k) {
      const double cx = integer(0, 7)*c, cy = integer(0, 7)*c, cz = integer(0, 7)*c;
      for (int m=0; m<5; ++m) {
        x.push_back(cx+uniform(0.1, 0.9)*c);
        y.push_back(cy+uniform(0.1, 0.9)*c);
        z.push_back(cz+uniform(0.1, 0.9)*c);
      }
    }
    randomPoints(qx, qy, qz, 100000, unit);
    Indices qi;
    for (int i=0; i<100000; ++i) qi.push_back(unsigned(integer(0, 299)));
    const MeshGrid g(x, y, z, c, opt);
    assert(g.get_cellsize() == c);
    checkLarge(g, qx, qy, qz, qi, 12.*c, 0.);
    checkLarge(g, qx, qy, qz, qi, inf, 0.3);
  }
  std::printf("  large batches, whole-cell rules:   ok\n");

  // A batch whose results run to millions of indices, with single queries of
  // 60000 results each among many small ones: 60000 objects packed in a cube
  // of side 0.02 inside a sparse background, and 60 queries (points and
  // objects) in that cube, spread through the batch, whose ball covers it.
  // However the batch stores results internally, this exceeds any buffer of
  // a few MiB, and single queries alone fill tens of thousands of entries.
  {
    std::vector<double> x, y, z, qx, qy, qz;
    randomPoints(x, y, z, 20000, unit);
    const Box blob = {{{0.49, 0.51}, {0.49, 0.51}, {0.49, 0.51}}};
    randomPoints(x, y, z, 60000, blob);
    randomPoints(qx, qy, qz, 120000, unit);
    Indices qi;
    for (int i=0; i<120000; ++i) qi.push_back(unsigned(integer(0, 19999)));
    for (int k=0; k<60; ++k) {
      const size_t at = size_t(integer(0, 119999));
      qx[at] = uniform(0.495, 0.505);
      qy[at] = uniform(0.495, 0.505);
      qz[at] = uniform(0.495, 0.505);
      qi[size_t(integer(0, 119999))] = unsigned(integer(20000, 79999));
    }
    const MeshGrid g(x, y, z, 0.01, MeshGrid::Options{unit, {}});
    checkLarge(g, qx, qy, qz, qi, 0.05, 0.);
    checkLarge(g, qx, qy, qz, qi, 0.05, 0.001);
  }
  std::printf("  large batches, very large results: ok\n");

  // Large index batches answered by cell. From the size at which a batch
  // answers in cell order, an index batch on a plain grid, with Rmax below
  // the radii from which whole cells are accepted or skipped, may answer all
  // the queries of one cell together; each query must still get exactly its
  // single query's count and indices, in order. The grid has holes and
  // relocated blocks: objects are removed, then added back, partly in the
  // clumps the batch queries most, and the added objects are queried too.
  // Indices repeat (some thousands of times) and come in random order, so a
  // cell's queries hold the same object more than once and are scattered
  // through the input. Half the objects are uniform, half in 40 tight clumps,
  // with exact duplicates, so cells hold from one query to thousands and
  // distances tie at 0. Radii run from 0 to 5 cell sides (below 6, where the
  // whole-cell rules start on a dense grid), as balls, shells, a thin shell
  // and Rmax = Rmin; the last batch holds a removed index and must raise.
  {
    const double c = 0.04;
    std::vector<double> x, y, z;
    randomPoints(x, y, z, 6000, unit);
    std::vector<std::array<double,3>> clumps;
    for (int k=0; k<40; ++k) clumps.push_back({uniform(0.05, 0.95), uniform(0.05, 0.95), uniform(0.05, 0.95)});
    std::normal_distribution<double> gauss(0., 0.3*c);
    for (int k=0; k<6000; ++k) {
      const auto& at = clumps[size_t(k % 40)];
      x.push_back(std::min(std::max(at[0]+gauss(rng), 0.), 1.));
      y.push_back(std::min(std::max(at[1]+gauss(rng), 0.), 1.));
      z.push_back(std::min(std::max(at[2]+gauss(rng), 0.), 1.));
    }
    for (int k=0; k<300; ++k) {
      const size_t from = size_t(integer(0, 11999));
      x.push_back(x[from]);
      y.push_back(y[from]);
      z.push_back(z[from]);
    }
    MeshGrid g(x, y, z, c, MeshGrid::Options{unit, {}});
    assert(g.get_cellsize() == c);
    std::vector<char> alive(x.size(), 1);
    for (int k=0; k<1500; ++k) {
      const unsigned int i = unsigned(integer(0, int(x.size())-1));
      if (alive[i]) {
        g.removeObject(i);
        alive[i] = 0;
      }
    }
    for (int k=0; k<1500; ++k) {
      const size_t from = size_t(integer(0, 11999));
      const double dx = (k % 2) ? 0. : uniform(-0.5, 0.5)*c;
      const unsigned int i = g.addObject(std::min(std::max(x[from]+dx, 0.), 1.), y[from], z[from]);
      assert(i == alive.size());
      alive.push_back(1);
    }
    Indices live;
    for (size_t i=0; i<alive.size(); ++i) if (alive[i]) live.push_back(unsigned(i));
    Indices qi;
    for (int k=0; k<110000; ++k) qi.push_back(live[size_t(integer(0, int(live.size())-1))]);
    for (int k=0; k<4000; ++k) qi[size_t(integer(0, 109999))] = live[k % 3];
    const std::vector<double> none;
    for (const auto& radii : {std::array<double,2>{0., 0.}, std::array<double,2>{0.5*c, 0.}, std::array<double,2>{c, 0.3*c},
                              std::array<double,2>{2.5*c, 0.}, std::array<double,2>{4.*c, 3.9*c},
                              std::array<double,2>{0.7*c, 0.7*c}})
      checkLarge(g, none, none, none, qi, radii[0], radii[1]);
    Counts countsi;
    for (const unsigned int i : qi) countsi.push_back(g.countObjects(i, 5.*c, 0.));
    assert(g.countObjects(qi, 5.*c) == countsi);
    Indices bad = qi;
    unsigned int removed = 0;
    while (alive[removed]) ++removed;
    bad[size_t(integer(0, 109999))] = removed;
    assert(raisesIndex([&] { g.closeObjects(bad, c); }));
    assert(raisesIndex([&] { g.countObjects(bad, c); }));
    bad.back() = unsigned(alive.size());
    assert(raisesIndex([&] { g.countObjects(bad, 2.*c, c); }));
  }
  std::printf("  large index batches, by cell:      ok\n");

  // The caller's container keeps its storage: a container-first call clears
  // it and refills it in place, so a buffer large enough for the result is
  // neither reallocated nor shrunk, whichever query fills it. Checked on the
  // plain and the periodic path, single and batch.
  for (const bool periodic : {false, true}) {
    MeshGrid::Options opt;
    opt.limits = unit;
    opt.periodic = {periodic, periodic, periodic};
    std::vector<double> x, y, z;
    randomPoints(x, y, z, 5000, unit);
    const MeshGrid g(x, y, z, 0.05, opt);
    Indices out;
    out.reserve(100000);
    const unsigned int* data = out.data();
    const size_t cap = out.capacity();
    for (int q=0; q<50; ++q) {
      g.closeObjects(x[q], y[q], z[q], 0.2, out);
      assert(out.data() == data && out.capacity() == cap && out == g.closeObjects(x[q], y[q], z[q], 0.2));
      g.closeObjects(unsigned(q), 0.3, 0.1, out);
      assert(out.data() == data && out.capacity() == cap && out == g.closeObjects(unsigned(q), 0.3, 0.1));
      g.nearestObjects(40, unsigned(q), out);
      assert(out.data() == data && out.capacity() == cap);
    }
    MeshGrid::Batch b;
    b.indices.reserve(1000000);
    b.offsets.reserve(1000);
    const unsigned int* bdata = b.indices.data();
    const std::uint64_t* odata = b.offsets.data();
    const std::vector<double> qx(x.begin(), x.begin()+200), qy(y.begin(), y.begin()+200), qz(z.begin(), z.begin()+200);
    for (int rep=0; rep<3; ++rep) {
      g.closeObjects(qx, qy, qz, 0.1+0.05*rep, b);
      assert(b.indices.data() == bdata && b.offsets.data() == odata);
    }
  }
  std::printf("  container storage reused: ok\n");

  // Const members may be called concurrently: several threads query one grid
  // at once, each with its own containers, and each must get exactly what a
  // single thread gets. Plain and periodic grids, every container-first form.
  for (const bool periodic : {false, true}) {
    MeshGrid::Options opt;
    opt.limits = unit;
    opt.periodic = {periodic, periodic, periodic};
    std::vector<double> x, y, z, qx, qy, qz;
    randomPoints(x, y, z, 20000, unit);
    randomPoints(qx, qy, qz, 3000, unit);
    Indices qi;
    for (unsigned int i=0; i<3000; ++i) qi.push_back(i*5);
    const MeshGrid g(x, y, z, 0.04, opt);

    struct Results { Indices single, knn, nn; MeshGrid::Batch close, closei; Counts count; };
    auto run = [&] (Results& r) {
      Indices one;
      for (size_t q=0; q<qx.size(); ++q) {
        g.closeObjects(qx[q], qy[q], qz[q], 0.06, 0.02, one);
        r.single.insert(r.single.end(), one.begin(), one.end());
      }
      g.nearestObjects(12, qx, qy, qz, r.knn);
      g.nearestObject(qi, r.nn);
      g.closeObjects(qx, qy, qz, 0.06, r.close);
      g.closeObjects(qi, 0.06, 0.01, r.closei);
      g.countObjects(qi, 0.08, r.count);
    };
    Results expect;
    run(expect);
    std::vector<Results> got(4);
    std::vector<std::thread> threads;
    for (auto& r : got) threads.emplace_back(run, std::ref(r));
    for (auto& t : threads) t.join();
    for (const auto& r : got)
      assert(r.single == expect.single && r.knn == expect.knn && r.nn == expect.nn &&
             r.close.indices == expect.close.indices && r.close.offsets == expect.close.offsets &&
             r.closei.indices == expect.closei.indices && r.closei.offsets == expect.closei.offsets &&
             r.count == expect.count);
  }
  std::printf("  concurrent queries on one grid: ok\n");

  // A grid holding one object: nearestObject by index has no answer, and the
  // batch says so even when empty.
  {
    const MeshGrid one({0.5}, {0.5}, {0.5}, 0.25, MeshGrid::Options{unit, {}});
    assert(raises([&] { one.nearestObject(Indices{}); }));
    assert(raises([&] { one.nearestObjects(1, Indices{}); }));
    assert(one.nearestObject(std::vector<double>{0.1}, {0.9}, {0.5}) == Indices{0});
    assert(one.countObjects(0u, inf) == 0);
    const MeshGrid none;
    assert(raises([&] { none.nearestObject(std::vector<double>{}, {}, {}); }));
    std::printf("  single object and empty grid: ok\n");
  }

  std::printf("all passed\n");
  return 0;
}
