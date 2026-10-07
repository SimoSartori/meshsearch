// A minimal consumer of meshsearch, built by the two projects next to it: one
// finds the installed package, the other includes the source tree. Each must
// compile against the header, link meshsearch::meshsearch and get the answer
// right, which is all a consumer's build can tell about the packaging.

#include <meshsearch/MeshGrid.h>

#include <cstdio>

int main ()
{
  const meshsearch::MeshGrid grid({0.1, 0.5, 0.9}, {0.1, 0.5, 0.9}, {0.1, 0.5, 0.9}, 0.25,
                                  {{0., 1.}, {0., 1.}, {0., 1.}});

  if (grid.nearestObject(0.45, 0.45, 0.45) != 1) {
    std::printf("wrong nearest object\n");
    return 1;
  }

  std::printf("meshsearch consumer: ok\n");
  return 0;
}
