// nanobind bindings for meshsearch::MeshGrid.
//
// The C++ names are unchanged; Python gets snake_case. Only what
// include/meshsearch/MeshGrid.h declares is exposed. The no-argument getters
// become read-only properties and the get_* spellings are not bound, the live
// object count is reached through len(), and index lists come back as numpy
// uint32 arrays.

#include "meshsearch/MeshGrid.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <optional>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

// Object indices are returned as uint32, which is only the C++ index type on a
// platform where unsigned int is 32 bits.
static_assert(sizeof(unsigned int) == 4, "object indices are exposed as uint32");

namespace {

  using IndexArray = nb::ndarray<nb::numpy, unsigned int, nb::ndim<1>>;
  using LimsArray = nb::ndarray<nb::numpy, double, nb::shape<3, 2>>;

  // Hand the vector's buffer to numpy and let a capsule own it, so the array
  // outlives the call without copying and without borrowing from the grid.
  IndexArray indexArray (std::vector<unsigned int>&& indices)
  {
    auto* held = new std::vector<unsigned int>(std::move(indices));
    nb::capsule owner(held, [](void* p) noexcept {
      delete static_cast<std::vector<unsigned int>*>(p);
    });
    return IndexArray(held->data(), {held->size()}, owner);
  }

  LimsArray limsArray (const std::vector<std::vector<double>>& lims)
  {
    auto* held = new std::array<double, 6>();
    for (size_t i=0; i<3; ++i) {
      (*held)[2*i] = lims[i][0];
      (*held)[2*i+1] = lims[i][1];
    }
    nb::capsule owner(held, [](void* p) noexcept {
      delete static_cast<std::array<double, 6>*>(p);
    });
    return LimsArray(held->data(), {3, 2}, owner);
  }

  nb::tuple triple (const std::array<unsigned int, 3>& values)
  {
    return nb::make_tuple(values[0], values[1], values[2]);
  }

}


NB_MODULE(meshsearch, m)
{
  m.doc() = "A uniform cubic mesh over a set of points, for neighbour queries.";

  // The C++ hierarchy is mirrored: meshsearch::IndexError derives from
  // meshsearch::Error, so meshsearch.IndexError derives from meshsearch.Error
  // as well as from Python's IndexError. Either base catches it, exactly as
  // catch (const Error&) and catch (const IndexError&) both do in C++.
  nb::object error_type = nb::exception<meshsearch::Error>(m, "Error", PyExc_ValueError);

  nb::object bases = nb::make_tuple(error_type, nb::borrow(PyExc_IndexError));
  PyObject* index_error = PyErr_NewException("meshsearch.IndexError", bases.ptr(), nullptr);
  if (index_error == nullptr) throw nb::python_error();
  m.attr("IndexError") = nb::steal(index_error);

  // Registered after Error's translator, because nanobind tries them in
  // reverse order and the derived C++ type has to be tried before its base.
  nb::register_exception_translator(
    [](const std::exception_ptr& p, void* payload) {
      try { std::rethrow_exception(p); }
      catch (const meshsearch::IndexError& e) {
        PyErr_SetString(static_cast<PyObject*>(payload), e.what());
      }
    },
    index_error);

  nb::class_<meshsearch::MeshGrid>(m, "MeshGrid",
      "Uniform cubic mesh over a set of points, for neighbour queries.\n\n"
      "Objects are identified by their position in the coordinate arrays.\n"
      "Indices are stable: remove_object leaves an index permanently invalid,\n"
      "add_object issues a new one, and indices are never reused.")

    .def("__init__",
         [](meshsearch::MeshGrid* self,
            const std::vector<double>& x,
            const std::vector<double>& y,
            const std::vector<double>& z,
            const double cellsize,
            const std::optional<std::vector<std::vector<double>>>& limits) {
           new (self) meshsearch::MeshGrid(
             x, y, z, cellsize,
             limits ? *limits : std::vector<std::vector<double>>());
         },
         "x"_a, "y"_a, "z"_a, "cellsize"_a, "limits"_a = nb::none(),
         "Build the grid over the points, which must be of equal length.\n"
         "limits is ((xlo,xhi),(ylo,yhi),(zlo,zhi)); when omitted, the data\n"
         "range padded by a margin proportional to it. Each side of the box\n"
         "is grown symmetrically to a whole number of cells.")

    .def("__len__", &meshsearch::MeshGrid::get_nObjects,
         "Number of objects in the grid, excluding removed ones.")

    .def("is_alive", &meshsearch::MeshGrid::isAlive, "index"_a,
         "True if index names an object still in the grid.")

    // The array owns its data through a capsule, so it must not be handed out
    // under the reference_internal policy a property would use by default.
    .def_prop_ro("lims",
                 [](const meshsearch::MeshGrid& self) { return limsArray(self.get_lims()); },
                 nb::rv_policy::move,
                 "The box, grown to whole cells, as a 3x2 array.")

    .def_prop_ro("n_cells",
                 [](const meshsearch::MeshGrid& self) { return triple(self.get_nCells()); },
                 "Cells per axis.")

    .def_prop_ro("cellsize", &meshsearch::MeshGrid::get_cellsize, "Cell side.")

    .def("nearest_object",
         nb::overload_cast<double, double, double>(&meshsearch::MeshGrid::nearestObject, nb::const_),
         "x"_a, "y"_a, "z"_a,
         "Index of the object nearest to a point.")

    .def("nearest_object",
         nb::overload_cast<unsigned int>(&meshsearch::MeshGrid::nearestObject, nb::const_),
         "index"_a,
         "Index of the object nearest to an object, which is excluded.")

    .def("nearest_objects",
         [](const meshsearch::MeshGrid& self, const unsigned int n,
            const double x, const double y, const double z) {
           return indexArray(self.nearestObjects(n, x, y, z));
         },
         "n"_a, "x"_a, "y"_a, "z"_a,
         "The n objects nearest to a point, nearest first.")

    .def("nearest_objects",
         [](const meshsearch::MeshGrid& self, const unsigned int n, const unsigned int index) {
           return indexArray(self.nearestObjects(n, index));
         },
         "n"_a, "index"_a,
         "The n objects nearest to an object, which is excluded, nearest first.")

    .def("close_objects",
         [](const meshsearch::MeshGrid& self, const double x, const double y, const double z,
            const double rmax, const double rmin) {
           return indexArray(self.closeObjects(x, y, z, rmax, rmin));
         },
         "x"_a, "y"_a, "z"_a, "rmax"_a, "rmin"_a = 0.,
         "Objects at distance d from a point, with rmin <= d <= rmax, in\n"
         "unspecified order.")

    .def("close_objects",
         [](const meshsearch::MeshGrid& self, const unsigned int index,
            const double rmax, const double rmin) {
           return indexArray(self.closeObjects(index, rmax, rmin));
         },
         "index"_a, "rmax"_a, "rmin"_a = 0.,
         "Objects at distance d from an object, with rmin <= d <= rmax, in\n"
         "unspecified order. The object itself is excluded.")

    .def("get_cell", &meshsearch::MeshGrid::get_Cell, "x"_a, "y"_a, "z"_a,
         "Linear index of the cell containing a point.")

    .def("get_cell_coords",
         [](const meshsearch::MeshGrid& self, const double x, const double y, const double z) {
           return triple(self.get_CellCoords(x, y, z));
         },
         "x"_a, "y"_a, "z"_a,
         "Cell coordinates (ix, iy, iz) of the cell containing a point.")

    .def("get_objects_in_cell",
         [](const meshsearch::MeshGrid& self, const double x, const double y, const double z) {
           return indexArray(self.get_ObjectsInCell(x, y, z));
         },
         "x"_a, "y"_a, "z"_a,
         "Objects in the cell containing a point.")

    .def("remove_object", &meshsearch::MeshGrid::removeObject, "index"_a,
         "Remove an object. Its index stays invalid for the life of the grid.")

    .def("add_object", &meshsearch::MeshGrid::addObject, "x"_a, "y"_a, "z"_a,
         "Add an object and return its index, one past the highest ever issued.");
}
