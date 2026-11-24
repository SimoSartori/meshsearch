// A periodic box with meshsearch::MeshGrid: neighbours are found across the
// faces of the box, as if it repeated forever in every direction.
//
//   cmake -S . -B build && cmake --build build
//   ./build/example_periodic

#include "meshsearch/MeshGrid.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

int main ()
{
  const size_t n = 5000;
  std::vector<double> X, Y, Z;
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> unit(0., 1.);
  for (size_t i=0; i<n; ++i) {
    X.push_back(unit(gen));
    Y.push_back(unit(gen));
    Z.push_back(unit(gen));
  }

  // Two objects on opposite sides of the x faces, 0.004 apart through the
  // boundary and 0.996 apart across the box.
  const unsigned int left = n;
  const unsigned int right = n+1;
  X.push_back(0.002); Y.push_back(0.5); Z.push_back(0.5);
  X.push_back(0.998); Y.push_back(0.5); Z.push_back(0.5);

  // A periodic axis needs its limits: they are the box, and the period is
  // hi-lo. The flags are per axis, so a slab periodic in x and y only would be
  // {true, true, false}.
  const double cellsize = 4./std::cbrt(double(n));
  meshsearch::MeshGrid::Options options;
  options.limits = std::array<std::array<double,2>,3>{{{0.,1.}, {0.,1.}, {0.,1.}}};
  options.periodic = {true, true, true};
  meshsearch::MeshGrid grid(X, Y, Z, cellsize, options);

  // Every cell has the same side, which divides the period exactly. It can be
  // a little smaller than the cellsize asked for, so read it back.
  printf("cellsize asked %.6f, used %.6f, %ux%ux%u cells\n",
         cellsize, grid.get_cellsize(),
         grid.get_nCells()[0], grid.get_nCells()[1], grid.get_nCells()[2]);

  // The nearest object to `left` is `right`, through the x boundary.
  const unsigned int nearest = grid.nearestObject(left);
  printf("nearest to object %u: object %u (expected %u)\n", left, nearest, right);

  // Radii are not limited by the box: +inf returns every object once, each
  // at its nearest image.
  const double inf = std::numeric_limits<double>::infinity();
  const std::vector<unsigned int> all = grid.closeObjects(0.5, 0.5, 0.5, inf);
  printf("objects within +inf of the centre: %zu of %u\n", all.size(), grid.get_nObjects());

  // A ball around a point at the corner takes in objects from all eight
  // corners of the box.
  const std::vector<unsigned int> corner = grid.closeObjects(0., 0., 0., 0.1);
  printf("objects within 0.1 of the corner (0, 0, 0): %zu\n", corner.size());

  return nearest == right && all.size() == grid.get_nObjects() ? 0 : 1;
}
