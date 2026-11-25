// Batches, container forms and counts with meshsearch::MeshGrid: many queries
// answered in one call, results written into containers the caller keeps, and
// neighbour counts without the neighbours.
//
//   cmake -S . -B build && cmake --build build
//   ./build/example_batch

#include "meshsearch/MeshGrid.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

int main ()
{
  const size_t n = 200000;
  std::vector<double> X, Y, Z;
  std::mt19937 gen(7);
  std::uniform_real_distribution<double> unit(0., 1.);
  for (size_t i=0; i<n; ++i) {
    X.push_back(unit(gen));
    Y.push_back(unit(gen));
    Z.push_back(unit(gen));
  }
  // The limits are the unit cube, so that every query point in it is
  // accepted. The cell side is two mean separations, where the mean
  // separation of n points in a volume V is cbrt(V/n).
  meshsearch::MeshGrid::Options options;
  options.limits = std::array<std::array<double,2>,3>{{{0.,1.}, {0.,1.}, {0.,1.}}};
  meshsearch::MeshGrid grid(X, Y, Z, 2.*std::cbrt(1./double(n)), options);

  std::vector<double> QX, QY, QZ;
  for (size_t i=0; i<50000; ++i) {
    QX.push_back(unit(gen));
    QY.push_back(unit(gen));
    QZ.push_back(unit(gen));
  }

  // A batch takes the query points as three vectors, and has the name of the
  // single query: vectors in place of numbers select it. Results come back in
  // query order. A batch is faster than the same queries asked one by one in
  // a loop, and a large one is answered internally in an
  // order that follows the cells, which is faster again.
  const std::vector<unsigned int> nearest = grid.nearestObject(QX, QY, QZ);
  printf("nearest object to query 0: %u\n", nearest[0]);

  // nearestObjects returns N per query, flat: query i's N at [i*N, (i+1)*N).
  const unsigned int N = 5;
  const std::vector<unsigned int> five = grid.nearestObjects(N, QX, QY, QZ);
  printf("five nearest to query 0:");
  for (unsigned int k=0; k<N; ++k) printf(" %u", five[k]);
  printf("\n");

  // closeObjects returns a varying number per query, so a batch comes back as
  // a MeshGrid::Batch: every index in one vector, and offsets with one more
  // entry than queries. Query i found indices[offsets[i]] up to, excluding,
  // indices[offsets[i+1]].
  const double radius = 0.02;
  const meshsearch::MeshGrid::Batch shells = grid.closeObjects(QX, QY, QZ, radius);
  const std::uint64_t first = shells.offsets[12];
  const std::uint64_t last = shells.offsets[13];
  printf("query 12 has %llu neighbours within %.2f\n", (unsigned long long)(last-first), radius);

  // countObjects gives the number closeObjects would return, without
  // collecting the indices, and is cheaper when the number is all that is
  // needed.
  const std::vector<std::uint64_t> within = grid.countObjects(QX, QY, QZ, radius);
  printf("query 12 counted: %llu\n", (unsigned long long)within[12]);

  // Every query that returns a container also has a form that writes into one
  // the caller passes as the last argument. It is cleared first and keeps its
  // capacity, so in a loop the memory is allocated once rather than once per
  // call. Declare it outside the loop.
  std::vector<unsigned int> found;
  size_t total = 0;
  for (unsigned int i=0; i<1000; ++i) {
    grid.closeObjects(i, radius, found);
    total += found.size();
  }
  printf("neighbours within %.2f of objects 0 to 999: %zu\n", radius, total);

  // The batch forms take a container in the same way: here a vector of
  // counts and a MeshGrid::Batch, each reused for three radii.
  std::vector<std::uint64_t> tally;
  meshsearch::MeshGrid::Batch rings;
  std::vector<std::uint64_t> totals;
  for (const double r : {0.01, 0.02, 0.04}) {
    grid.countObjects(QX, QY, QZ, r, tally);
    grid.closeObjects(QX, QY, QZ, r, rings);
    totals.push_back(std::accumulate(tally.begin(), tally.end(), std::uint64_t(0)));
    if (rings.indices.size() != totals.back()) return 1;
  }
  printf("neighbours over all queries within 0.01, 0.02, 0.04: %llu %llu %llu\n",
         (unsigned long long)totals[0], (unsigned long long)totals[1], (unsigned long long)totals[2]);

  // Batches also take object indices; each object is excluded from its own
  // answer.
  const std::vector<unsigned int> objects = {0, 1, 2};
  const std::vector<std::uint64_t> around = grid.countObjects(objects, radius);
  printf("neighbours of objects 0, 1, 2: %llu %llu %llu\n",
         (unsigned long long)around[0], (unsigned long long)around[1], (unsigned long long)around[2]);

  // contains and isAlive answer for many points or indices at once, and never
  // raise on the values: one std::uint8_t, 0 or 1, per entry. contains is
  // whether a query at the point would be accepted, that is, whether it lies
  // in the box, so a batch can be screened before it is asked. isAlive is the
  // one member taking an index that does not raise: an index out of range or
  // removed is 0.
  std::vector<double> screened = QX;
  screened[99] = 5.;
  const std::vector<std::uint8_t> accepted = grid.contains(screened, QY, QZ);
  const std::vector<unsigned int> asked = {0, 1, (unsigned int)n};
  const std::vector<std::uint8_t> alive = grid.isAlive(asked);
  printf("query 99 lies in the box: %d; objects 0, 1 and n are alive: %d %d %d\n",
         accepted[99], alive[0], alive[1], alive[2]);

  // A batch raises Error if any query does, for example a point outside the
  // box. Nothing partial is returned, and a container passed in is then left
  // with unspecified content.
  std::vector<double> bad = QX;
  bad[99] = 5.;
  bool raised = false;
  try {
    grid.countObjects(bad, QY, QZ, radius, tally);
  }
  catch (const meshsearch::Error& e) {
    raised = true;
    printf("a query point outside the box raised Error: %s\n", e.what());
  }

  return raised && last-first == within[12] && !accepted[99] && accepted[98] && alive[0] && !alive[2] ? 0 : 1;
}
