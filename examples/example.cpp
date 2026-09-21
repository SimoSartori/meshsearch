// A short tour of meshsearch::MeshGrid: build a grid over a set of points,
// ask the three kinds of neighbour question, modify the grid, and handle a
// bad query.
//
//   cmake -S . -B build && cmake --build build
//   ./build/example

#include "meshsearch/MeshGrid.h"

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

  // The cell side sets the cost of every query: too large and each cell holds
  // many objects to test, too small and a query walks many cells. The mean
  // separation is a reasonable starting point. Four time the mean separation 
  // is optimal in most cases.
  //
  // Coordinates, cellsize and radii must share one unit. The library does not
  // impose one, so whatever the points are in, the radii are in too.
  const double cellsize = 4./std::cbrt(double(n));

  // With no limits given, the box is fitted to the data and then grown
  // symmetrically to a whole number of cells, so get_lims() can be slightly
  // wider than the points. Pass limits explicitly to control it:
  //   MeshGrid grid(X, Y, Z, cellsize, {{0.,1.}, {0.,1.}, {0.,1.}});
  meshsearch::MeshGrid grid(X, Y, Z, cellsize);

  const auto lims = grid.get_lims();
  const auto nCells = grid.get_nCells();
  printf("%u objects, %ux%ux%u cells of %.4f\n",
         grid.get_nObjects(), nCells[0], nCells[1], nCells[2], grid.get_cellsize());
  printf("box x [%.4f, %.4f]  y [%.4f, %.4f]  z [%.4f, %.4f]\n\n",
         lims[0][0], lims[0][1], lims[1][0], lims[1][1], lims[2][0], lims[2][1]);

  // 1. The object nearest a point.
  const unsigned int nearest = grid.nearestObject(0.5, 0.5, 0.5);
  printf("nearest to the box centre: object %u at (%.4f, %.4f, %.4f)\n",
         nearest, X[nearest], Y[nearest], Z[nearest]);

  // 2. The N nearest, returned nearest first.
  const std::vector<unsigned int> ten = grid.nearestObjects(10, 0.5, 0.5, 0.5);
  printf("ten nearest to the centre:");
  for (const auto index : ten) printf(" %u", index);
  printf("\n");

  // 3. Every object in a spherical shell, Rmin <= d <= Rmax, in no particular
  //    order. Rmin defaults to 0, giving a ball. A shell reaching outside the
  //    box is truncated rather than rejected.
  const std::vector<unsigned int> shell = grid.closeObjects(0.5, 0.5, 0.5, 0.15, 0.10);
  printf("objects between 0.10 and 0.15 of the centre: %zu\n", shell.size());

  // Each query also has an overload taking an object already in the grid,
  // which is then excluded from its own answer.
  const std::vector<unsigned int> around = grid.nearestObjects(5, nearest);
  printf("five nearest to object %u:", nearest);
  for (const auto index : around) printf(" %u", index);
  printf("\n\n");

  // Const members may be called concurrently on one grid. The two below may
  // not, and must not run alongside a query.
  //
  // Indices are stable: removing an object leaves every other index valid and
  // its own permanently invalid, and adding issues a fresh one. Indices are
  // never reused, so a stored index never comes to mean a different object.
  grid.removeObject(nearest);
  printf("removed object %u: %u objects left, isAlive(%u) = %s\n",
         nearest, grid.get_nObjects(), nearest, grid.isAlive(nearest) ? "true" : "false");

  const unsigned int added = grid.addObject(0.5, 0.5, 0.5);
  printf("added an object at the centre: index %u, %u objects\n",
         added, grid.get_nObjects());
  printf("nearest to the centre is now object %u\n\n", grid.nearestObject(0.5, 0.5, 0.5));

  // Errors are exceptions, never return codes. A point outside the box, a
  // negative or NaN radius, or a count larger than the grid holds raises
  // Error; naming an index that is out of range or removed raises IndexError,
  // which derives from Error, so catching Error alone catches both.
  try {
    grid.nearestObject(nearest);
  }
  catch (const meshsearch::IndexError& e) {
    printf("querying the removed object raised IndexError: %s\n", e.what());
  }

  try {
    grid.closeObjects(2., 2., 2., 0.1);
  }
  catch (const meshsearch::Error& e) {
    printf("querying a point outside the box raised Error: %s\n", e.what());
  }

  return 0;
}
