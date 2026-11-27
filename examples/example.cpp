// A short tour of meshsearch::MeshGrid: build a grid over a set of points,
// ask the four neighbour queries, modify the grid, and handle a bad query.
//
//   cmake -S . -B build && cmake --build build
//   ./build/example

#include "meshsearch/MeshGrid.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main ()
{
  // The points. Coordinates are passed as three parallel vectors and copied
  // into the grid; an object is identified by its position in them.
  const size_t n = 5000;
  std::vector<double> X, Y, Z;
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> unit(0., 1.);
  for (size_t i=0; i<n; ++i) {
    X.push_back(unit(gen));
    Y.push_back(unit(gen));
    Z.push_back(unit(gen));
  }

  // The box. Give the limits of the region the points and every later query
  // point belong to, here the unit cube. Without limits the box is fitted to
  // the points, and a query point outside their range raises Error.
  meshsearch::MeshGrid::Options options;
  options.limits = std::array<std::array<double,2>,3>{{{0.,1.}, {0.,1.}, {0.,1.}}};

  // The cell side, the one tuning parameter. The mean separation of n points
  // in a volume V is cbrt(V/n). On uniform data the guide advises starting
  // between 1.5 and 2.5 mean separations, here two, and shows what other
  // choices cost. Coordinates, cell side and radii share one unit, whatever
  // it is.
  const double volume = 1.;
  const double separation = std::cbrt(volume/double(n));
  const double cellsize = 2.*separation;

  meshsearch::MeshGrid grid(X, Y, Z, cellsize, options);

  const auto lims = grid.get_lims();
  const auto nCells = grid.get_nCells();
  printf("%u objects, %ux%ux%u cells of %.4f\n",
         grid.get_nObjects(), nCells[0], nCells[1], nCells[2], grid.get_cellsize());
  printf("box x [%.4f, %.4f]  y [%.4f, %.4f]  z [%.4f, %.4f]\n\n",
         lims[0][0], lims[0][1], lims[1][0], lims[1][1], lims[2][0], lims[2][1]);

  // The same points as one array of interleaved coordinates, x0 y0 z0 x1 y1
  // z1 ..., given by a pointer and the number of points. The grid copies
  // from it and keeps no reference, so the array can go as soon as the
  // constructor returns. Three pointers to separate arrays work as well.
  // Either form builds the same grid, with the same results in the same
  // order.
  std::vector<double> XYZ;
  for (size_t i=0; i<n; ++i) {
    XYZ.push_back(X[i]);
    XYZ.push_back(Y[i]);
    XYZ.push_back(Z[i]);
  }
  const meshsearch::MeshGrid interleaved(XYZ.data(), n, cellsize, options);
  const bool same = interleaved.get_nCells() == nCells
    && interleaved.nearestObjects(10, 0.5, 0.5, 0.5) == grid.nearestObjects(10, 0.5, 0.5, 0.5)
    && interleaved.closeObjects(0.5, 0.5, 0.5, 0.15, 0.10) == grid.closeObjects(0.5, 0.5, 0.5, 0.15, 0.10);
  printf("the interleaved array builds the same grid: %s\n\n", same ? "yes" : "no");

  const unsigned int nearest = grid.nearestObject(0.5, 0.5, 0.5);
  printf("nearest to the box centre: object %u at (%.4f, %.4f, %.4f)\n",
         nearest, X[nearest], Y[nearest], Z[nearest]);

  const std::vector<unsigned int> ten = grid.nearestObjects(10, 0.5, 0.5, 0.5);
  printf("ten nearest to the centre:");
  for (const auto index : ten) printf(" %u", index);
  printf("\n");

  // A shell is Rmin <= d <= Rmax, both ends included; with Rmin left out it
  // is a ball. The order of the indices is not meaningful.
  const std::vector<unsigned int> shell = grid.closeObjects(0.5, 0.5, 0.5, 0.15, 0.10);
  const size_t counted = grid.countObjects(0.5, 0.5, 0.5, 0.15, 0.10);
  printf("objects between 0.10 and 0.15 of the centre: %zu (counted: %zu)\n", shell.size(), counted);

  // Asked about an object in the grid, a query leaves that object out of its
  // own answer.
  const std::vector<unsigned int> around = grid.nearestObjects(5, nearest);
  printf("five nearest to object %u:", nearest);
  for (const auto index : around) printf(" %u", index);
  printf("\n\n");

  // Removing an object leaves its index permanently invalid; adding one
  // issues a new index. No index is ever reused. Neither call may run while
  // another thread uses the grid.
  grid.removeObject(nearest);
  printf("removed object %u: %u objects left, isAlive(%u) = %s\n",
         nearest, grid.get_nObjects(), nearest, grid.isAlive(nearest) ? "true" : "false");

  const unsigned int added = grid.addObject(0.5, 0.5, 0.5);
  printf("added an object at the centre: index %u, %u objects\n",
         added, grid.get_nObjects());
  printf("nearest to the centre is now object %u\n\n", grid.nearestObject(0.5, 0.5, 0.5));

  // IndexError derives from Error, so catching Error alone catches both.
  try {
    grid.nearestObject(nearest);
  }
  catch (const meshsearch::IndexError& e) {
    printf("querying the removed object raised IndexError: %s\n", e.what());
  }

  // A point outside the box is an error, never a silent empty answer;
  // contains() tells beforehand.
  printf("contains(2, 2, 2) = %s\n", grid.contains(2., 2., 2.) ? "true" : "false");
  try {
    grid.closeObjects(2., 2., 2., 0.1);
  }
  catch (const meshsearch::Error& e) {
    printf("querying a point outside the box raised Error: %s\n", e.what());
  }

  return same && shell.size() == counted && ten.size() == 10 ? 0 : 1;
}
