// Tests for meshsearch::Workers, the last argument of every batch form.
//
// The header guarantees that results do not depend on the number of
// workers: the same indices and counts, in the same order, for any count.
// test_forms establishes that a batch with one worker equals the single
// queries; here every batch form with 2, 3, 7 and -1 workers is compared,
// element by element, with the same batch with one worker. A count above the
// number of cores (7 on a small machine, or -1 on a large one) is allowed and
// must make no difference either.
//
// A batch may run single-threaded when it is small, so the batches here hold
// a few thousand queries, enough for every worker to get some; the batches of 100000 queries and more, which
// are answered in cell order or grouped by cell, are compared with several
// workers in test_forms (checkLarge and the first large-batch group).
//
// Also checked: the values Workers accepts and refuses; that a batch raises
// the same error with any number of workers when several queries are
// invalid (inputs are checked before any worker starts); that an exception
// thrown inside a worker reaches the caller instead of terminating the
// program, and leaves the grid usable.
//
// NDEBUG is undefined below so that assert() stays live whatever the build
// type.

#undef NDEBUG

// The replaced operator new and delete below both use malloc and free; GCC,
// seeing free reached from a pointer it knows came from operator new, warns
// about a mismatch that is not there.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

#include "meshsearch/MeshGrid.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <random>
#include <string>
#include <thread>
#include <vector>

using meshsearch::Error;
using meshsearch::IndexError;
using meshsearch::MeshGrid;
using meshsearch::Workers;

// ------------------------------------------------------------------
// An allocation that fails in any thread but the one that armed it. A batch
// allocates inside its workers (each worker's result storage, its query
// buffers), so arming this makes a worker throw std::bad_alloc from within a
// running batch, which no input can provoke otherwise.
// ------------------------------------------------------------------

namespace {
  std::atomic<bool> armed(false);
  std::thread::id armer;
}

void* operator new (std::size_t size)
{
  if (armed.load() && std::this_thread::get_id() != armer) throw std::bad_alloc();
  if (void* p = std::malloc(size == 0 ? 1 : size)) return p;
  throw std::bad_alloc();
}

void operator delete (void* p) noexcept { std::free(p); }

void operator delete (void* p, std::size_t) noexcept { std::free(p); }

namespace {

  using Box = std::array<std::array<double,2>,3>;
  using Indices = std::vector<unsigned int>;
  using Doubles = std::vector<double>;
  using Counts = std::vector<std::uint64_t>;
  using Batch = MeshGrid::Batch;

  const double inf = std::numeric_limits<double>::infinity();
  const Box unit = {{{0., 1.}, {0., 1.}, {0., 1.}}};

  std::mt19937_64 rng(20261127);

  double uniform (const double a, const double b) { return std::uniform_real_distribution<double>(a, b)(rng); }
  int integer (const int a, const int b) { return std::uniform_int_distribution<int>(a, b)(rng); }

  const int counts[] = {2, 3, 7, -1};

  bool equal (const Batch& a, const Batch& b) { return a.indices == b.indices && a.offsets == b.offsets; }

  Indices stale () { return Indices(7, 12345u); }
  Batch staleBatch () { return Batch{stale(), Counts(3, 99)}; }

  void randomPoints (Doubles& x, Doubles& y, Doubles& z, const int n, const Box& box)
  {
    for (int i=0; i<n; ++i) {
      x.push_back(uniform(box[0][0], box[0][1]));
      y.push_back(uniform(box[1][0], box[1][1]));
      z.push_back(uniform(box[2][0], box[2][1]));
    }
  }

  Indices randomLive (const MeshGrid& g, const unsigned int issued, const int n)
  {
    Indices live, qi;
    for (unsigned int i=0; i<issued; ++i) if (g.isAlive(i)) live.push_back(i);
    for (int k=0; k<n; ++k) qi.push_back(live[size_t(integer(0, int(live.size())-1))]);
    return qi;
  }

  // ------------------------------------------------------------------
  // Every batch form, returning and into a container, with and without Rmin,
  // and the index forms answered into their own input, with several workers
  // against one worker.
  // ------------------------------------------------------------------

  void checkAll (const MeshGrid& g, const Doubles& qx, const Doubles& qy, const Doubles& qz, const Indices& qi,
                 const double R, const double Rmin, const unsigned int N)
  {
    const unsigned int n = g.get_nObjects();
    const unsigned int Ni = std::min(N, n-1);
    const Indices nn = g.nearestObject(qx, qy, qz);
    const Indices nni = g.nearestObject(qi);
    const Indices knn = g.nearestObjects(N, qx, qy, qz);
    const Indices knni = g.nearestObjects(Ni, qi);
    const Batch close = g.closeObjects(qx, qy, qz, R, Rmin);
    const Batch closei = g.closeObjects(qi, R, Rmin);
    const Batch ball = g.closeObjects(qx, qy, qz, R);
    const Batch balli = g.closeObjects(qi, R);
    const Counts count = g.countObjects(qx, qy, qz, R, Rmin);
    const Counts counti = g.countObjects(qi, R, Rmin);
    const Counts countBall = g.countObjects(qx, qy, qz, R);
    const Counts countBalli = g.countObjects(qi, R);

    for (const int w : counts) {
      const Workers W(w);
      Indices out = stale();
      assert(g.nearestObject(qx, qy, qz, W) == nn);
      g.nearestObject(qx, qy, qz, out, W);
      assert(out == nn);
      assert(g.nearestObject(qi, W) == nni);
      out = stale();
      g.nearestObject(qi, out, W);
      assert(out == nni);
      Indices same = qi;
      g.nearestObject(same, same, W);
      assert(same == nni);

      assert(g.nearestObjects(N, qx, qy, qz, W) == knn);
      out = stale();
      g.nearestObjects(N, qx, qy, qz, out, W);
      assert(out == knn);
      assert(g.nearestObjects(Ni, qi, W) == knni);
      out = stale();
      g.nearestObjects(Ni, qi, out, W);
      assert(out == knni);
      same = qi;
      g.nearestObjects(Ni, same, same, W);
      assert(same == knni);

      assert(equal(g.closeObjects(qx, qy, qz, R, Rmin, W), close));
      Batch b = staleBatch();
      g.closeObjects(qx, qy, qz, R, Rmin, b, W);
      assert(equal(b, close));
      b = staleBatch();
      g.closeObjects(qx, qy, qz, R, b, W);
      assert(equal(b, ball));
      assert(equal(g.closeObjects(qi, R, Rmin, W), closei));
      b = staleBatch();
      g.closeObjects(qi, R, Rmin, b, W);
      assert(equal(b, closei));
      b = staleBatch();
      g.closeObjects(qi, R, b, W);
      assert(equal(b, balli));
      Batch own{qi, Counts(3, 99)};
      g.closeObjects(own.indices, R, Rmin, own, W);
      assert(equal(own, closei));

      assert(g.countObjects(qx, qy, qz, R, Rmin, W) == count);
      Counts c(5, 99);
      g.countObjects(qx, qy, qz, R, Rmin, c, W);
      assert(c == count);
      c.assign(5, 99);
      g.countObjects(qx, qy, qz, R, c, W);
      assert(c == countBall);
      assert(g.countObjects(qi, R, Rmin, W) == counti);
      c.assign(5, 99);
      g.countObjects(qi, R, Rmin, c, W);
      assert(c == counti);
      c.assign(5, 99);
      g.countObjects(qi, R, c, W);
      assert(c == countBalli);
    }
  }

  // ------------------------------------------------------------------
  // Errors: whichever worker reaches an invalid query first, the batch raises
  // what it raises with one worker, which is the error of the first invalid
  // query in input order. The message tells the errors apart.
  // ------------------------------------------------------------------

  template <class F>
  std::string message (F&& f)
  {
    try { f(); }
    catch (const IndexError& e) { return std::string("IndexError: ")+e.what(); }
    catch (const Error& e) { return std::string("Error: ")+e.what(); }
    return "no error";
  }

  void checkErrors (const MeshGrid& g, const Doubles& qx, const Doubles& qy, const Doubles& qz, const Indices& qi,
                    const unsigned int removed)
  {
    const auto box = g.get_lims();
    Doubles bx = qx, by = qy, bz = qz;
    bz[qz.size()/3] = box[2][1]+1.;
    by[qy.size()/2] = box[1][0]-1.;
    bx[qx.size()-2] = std::nan("");
    Indices bi = qi;
    bi[qi.size()/4] = removed;
    bi[qi.size()/2] = std::numeric_limits<unsigned int>::max();
    Indices bj = qi;
    bj[qi.size()/5] = g.get_nObjects()+10000;
    bj[qi.size()-1] = removed;

    const std::string point[] = {
      message([&] { g.nearestObject(bx, by, bz); }),
      message([&] { g.nearestObjects(3, bx, by, bz); }),
      message([&] { g.closeObjects(bx, by, bz, 0.1); }),
      message([&] { g.countObjects(bx, by, bz, 0.1); })};
    for (const std::string& m : point) assert(m == "Error: Z coordinate outside the box");
    const std::string index[] = {
      message([&] { g.nearestObject(bi); }),
      message([&] { g.nearestObjects(3, bi); }),
      message([&] { g.closeObjects(bi, 0.1); }),
      message([&] { g.countObjects(bi, 0.1); })};
    for (const std::string& m : index) assert(m == "IndexError: object index has been removed");
    assert(message([&] { g.closeObjects(bj, 0.1); }) == "IndexError: object index out of range");

    for (const int w : counts) {
      const Workers W(w);
      assert(message([&] { g.nearestObject(bx, by, bz, W); }) == point[0]);
      assert(message([&] { g.nearestObjects(3, bx, by, bz, W); }) == point[1]);
      assert(message([&] { g.closeObjects(bx, by, bz, 0.1, 0., W); }) == point[2]);
      assert(message([&] { g.countObjects(bx, by, bz, 0.1, 0., W); }) == point[3]);
      assert(message([&] { g.nearestObject(bi, W); }) == index[0]);
      assert(message([&] { g.nearestObjects(3, bi, W); }) == index[1]);
      assert(message([&] { g.closeObjects(bi, 0.1, 0., W); }) == index[2]);
      assert(message([&] { g.countObjects(bi, 0.1, 0., W); }) == index[3]);
      assert(message([&] { g.closeObjects(bj, 0.1, 0., W); }) == "IndexError: object index out of range");
      assert(message([&] { g.countObjects(bj, 0.1, 0., W); }) == "IndexError: object index out of range");
      assert(message([&] { g.closeObjects(qi, -1., 0., W); }) == "Error: Rmax < 0");
      assert(message([&] { g.nearestObjects(g.get_nObjects(), qi, W); }) ==
             "Error: N is larger than the number of available neighbours");
    }
  }

  // ------------------------------------------------------------------
  // An exception thrown inside a worker is rethrown in the caller, after all
  // workers have stopped; the program does not terminate, and the grid and
  // the caller's containers are usable afterwards. The calling thread takes
  // part in the work, so on a busy machine it may finish a batch before any
  // other worker has allocated; such a call must then simply be right, and it
  // is repeated until a worker does throw.
  // ------------------------------------------------------------------

  template <class F>
  bool badAlloc (F&& f)
  {
    armer = std::this_thread::get_id();
    armed.store(true);
    bool thrown = false;
    try { f(); }
    catch (const std::bad_alloc&) { thrown = true; }
    armed.store(false);
    return thrown;
  }

  template <class F, class Check>
  void failsInWorker (F&& f, Check&& check)
  {
    for (int attempt=0; attempt<100; ++attempt) {
      if (badAlloc(f)) return;
      check();
    }
    assert(!"no worker ever threw");
  }

  void checkWorkerFailure (const MeshGrid& g, const Doubles& qx, const Doubles& qy, const Doubles& qz,
                           const Indices& qi, const double R)
  {
    const Batch close = g.closeObjects(qx, qy, qz, R);
    const Batch closei = g.closeObjects(qi, R);
    const Indices knn = g.nearestObjects(5, qx, qy, qz);
    Batch out;
    Indices kout;
    for (const int w : {2, 7}) {
      const Workers W(w);
      failsInWorker([&] { g.closeObjects(qx, qy, qz, R, out, W); }, [&] { assert(equal(out, close)); });
      failsInWorker([&] { g.closeObjects(qi, R, out, W); }, [&] { assert(equal(out, closei)); });
      failsInWorker([&] { g.nearestObjects(5, qx, qy, qz, kout, W); }, [&] { assert(kout == knn); });
      g.closeObjects(qx, qy, qz, R, out, W);
      assert(equal(out, close));
      g.closeObjects(qi, R, out, W);
      assert(equal(out, closei));
      g.nearestObjects(5, qx, qy, qz, kout, W);
      assert(kout == knn);
    }
  }

}


int main ()
{
  std::printf("batch forms with several workers against one worker\n");

  // Workers takes 1 or more, or -1 for every core; anything else raises.
  // More workers than cores is allowed.
  assert(Workers().get_count() == 1);
  assert(Workers(1).get_count() == 1);
  assert(Workers(-1).get_count() == -1);
  assert(Workers(4096).get_count() == 4096);
  for (const int bad : {0, -2, -3, INT_MIN}) {
    bool raised = false;
    try { Workers w(bad); (void)w; }
    catch (const Error&) { raised = true; }
    assert(raised);
  }
  std::printf("  values accepted and refused: ok\n");

  // Plain grids, dense and sparse: walks reaching far shells, Rmin > 0, an
  // infinite radius. 4000 queries of each form, below the size from which a
  // batch answers in cell order, so this checks the path in input order.
  for (const auto& regime : {std::array<double,2>{20000., 0.03}, std::array<double,2>{300., 0.08}}) {
    Doubles x, y, z, qx, qy, qz;
    randomPoints(x, y, z, int(regime[0]), unit);
    const MeshGrid g(x, y, z, regime[1], MeshGrid::Options{unit, {}});
    randomPoints(qx, qy, qz, 4000, unit);
    const Indices qi = randomLive(g, g.get_nObjects(), 4000);
    checkAll(g, qx, qy, qz, qi, 0.05, 0.02, 8);
    checkAll(g, qx, qy, qz, qi, 0.1, 0., 1);
    if (regime[0] < 1000.) checkAll(g, qx, qy, qz, qi, inf, 0.5, 40);
  }
  std::printf("  plain grids, dense and sparse: ok\n");

  // Periodic grids, every axis and one axis: queries near the faces see
  // neighbours through the wrap.
  for (const auto& per : {std::array<bool,3>{true, true, true}, std::array<bool,3>{false, true, false}}) {
    MeshGrid::Options opt;
    opt.limits = Box{{{-2., 2.}, {0., 8.}, {10., 14.}}};
    opt.periodic = per;
    Doubles x, y, z, qx, qy, qz;
    randomPoints(x, y, z, 8000, *opt.limits);
    const MeshGrid g(x, y, z, 0.37, opt);
    randomPoints(qx, qy, qz, 4000, *opt.limits);
    const Indices qi = randomLive(g, g.get_nObjects(), 4000);
    checkAll(g, qx, qy, qz, qi, 0.9, 0.2, 10);
    checkAll(g, qx, qy, qz, qi, 1.2, 0., 3);
  }
  std::printf("  periodic grids: ok\n");

  // Whole cells accepted or skipped without measuring each object: a dense
  // grid of clumps (5 objects per occupied cell) and radii of 12 cell sides
  // and +inf with Rmin, plain and periodic (as in test_forms).
  for (const bool periodic : {false, true}) {
    MeshGrid::Options opt;
    opt.limits = unit;
    opt.periodic = {periodic, periodic, periodic};
    const double c = 1./8.;
    Doubles x, y, z, qx, qy, qz;
    for (int k=0; k<60; ++k) {
      const double cx = integer(0, 7)*c, cy = integer(0, 7)*c, cz = integer(0, 7)*c;
      for (int m=0; m<5; ++m) {
        x.push_back(cx+uniform(0.1, 0.9)*c);
        y.push_back(cy+uniform(0.1, 0.9)*c);
        z.push_back(cz+uniform(0.1, 0.9)*c);
      }
    }
    const MeshGrid g(x, y, z, c, opt);
    randomPoints(qx, qy, qz, 4000, unit);
    const Indices qi = randomLive(g, g.get_nObjects(), 4000);
    checkAll(g, qx, qy, qz, qi, 12.*c, 0., 4);
    checkAll(g, qx, qy, qz, qi, inf, 0.3, 4);
  }
  std::printf("  whole-cell rules: ok\n");

  // An integer lattice with coincident objects, removals and additions:
  // distances tie in large groups, radii equal lattice distances, live
  // indices have gaps and cells hold moved blocks. Then the errors, with a
  // removed index, and the exception from a worker.
  {
    Doubles x, y, z;
    for (int i=0; i<12; ++i) for (int j=0; j<12; ++j) for (int k=0; k<12; ++k) {
      x.push_back(i); y.push_back(j); z.push_back(k);
    }
    for (int i=0; i<30; ++i) { x.push_back(4.); y.push_back(4.); z.push_back(4.); }
    MeshGrid g(x, y, z, 1., MeshGrid::Options{Box{{{0., 11.}, {0., 11.}, {0., 11.}}}, {}});
    for (unsigned int i=0; i<x.size(); i+=7) g.removeObject(i);
    for (int i=0; i<300; ++i) g.addObject(integer(0, 11), integer(0, 11), integer(0, 11));
    const unsigned int issued = unsigned(x.size())+300;
    Doubles qx, qy, qz;
    for (int i=0; i<4000; ++i) {
      qx.push_back(integer(0, 22)*0.5);
      qy.push_back(integer(0, 22)*0.5);
      qz.push_back(integer(0, 22)*0.5);
    }
    const Indices qi = randomLive(g, issued, 4000);
    checkAll(g, qx, qy, qz, qi, std::sqrt(2.), 1., 20);
    checkAll(g, qx, qy, qz, qi, 3., 0., 7);
    checkErrors(g, qx, qy, qz, qi, 0);
    checkWorkerFailure(g, qx, qy, qz, qi, 2.);
  }
  std::printf("  lattice, ties, edits, errors, worker failure: ok\n");

  // From 100000 queries a batch answers in cell order, and an index batch on
  // a plain grid below the whole-cell radii is grouped by cell: the errors and
  // the exception from a worker on those paths too.
  {
    Doubles x, y, z, qx, qy, qz;
    randomPoints(x, y, z, 20000, unit);
    MeshGrid g(x, y, z, 0.04, MeshGrid::Options{unit, {}});
    g.removeObject(17);
    randomPoints(qx, qy, qz, 110000, unit);
    const Indices qi = randomLive(g, 20000, 110000);
    checkErrors(g, qx, qy, qz, qi, 17);
    checkWorkerFailure(g, qx, qy, qz, qi, 0.03);
  }
  std::printf("  cell order and by cell, errors, worker failure: ok\n");

  std::printf("all passed\n");
  return 0;
}
