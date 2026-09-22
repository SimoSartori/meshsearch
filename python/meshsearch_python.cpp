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

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <string>
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

  // Query points for the batched calls. A strict 1-D contiguous double array,
  // so that an array argument matches this and a scalar cannot, which is what
  // keeps the overload dispatch unambiguous.
  using Coords = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

  using IndexArray2D = nb::ndarray<nb::numpy, unsigned int, nb::ndim<2>>;

  IndexArray2D indexArray2D (std::vector<unsigned int>&& values,
                             const size_t rows, const size_t cols)
  {
    auto* held = new std::vector<unsigned int>(std::move(values));
    nb::capsule owner(held, [](void* p) noexcept {
      delete static_cast<std::vector<unsigned int>*>(p);
    });
    return IndexArray2D(held->data(), {rows, cols}, owner);
  }

  size_t batchSize (const Coords& x, const Coords& y, const Coords& z)
  {
    if (x.shape(0) != y.shape(0) || x.shape(0) != z.shape(0))
      throw meshsearch::Error("the query coordinate arrays differ in length");
    return x.shape(0);
  }

  // A batched call fails where the single-point call would, naming the query
  // that did it, and returns nothing partial.
  std::string queryPrefix (const size_t i)
  {
    return "query "+std::to_string(i)+": ";
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
         [](const meshsearch::MeshGrid& self, const Coords& x, const Coords& y, const Coords& z) {
           const size_t n = batchSize(x, y, z);
           const double* px = x.data();
           const double* py = y.data();
           const double* pz = z.data();
           std::vector<unsigned int> out(n);
           {
             nb::gil_scoped_release released;
             for (size_t i=0; i<n; ++i) {
               try { out[i] = self.nearestObject(px[i], py[i], pz[i]); }
               catch (const meshsearch::IndexError& e) { throw meshsearch::IndexError(queryPrefix(i)+e.what()); }
               catch (const meshsearch::Error& e) { throw meshsearch::Error(queryPrefix(i)+e.what()); }
             }
           }
           return indexArray(std::move(out));
         },
         "x"_a, "y"_a, "z"_a,
         "The object nearest to each of an array of points: one index per\n"
         "query, in query order.")

    .def("nearest_object",
         nb::overload_cast<double, double, double>(&meshsearch::MeshGrid::nearestObject, nb::const_),
         "x"_a, "y"_a, "z"_a,
         "Index of the object nearest to a point. An object lying on the\n"
         "point is at distance 0 and is the answer.")

    .def("nearest_object",
         nb::overload_cast<unsigned int>(&meshsearch::MeshGrid::nearestObject, nb::const_),
         "index"_a,
         "Index of the object nearest to an object. The query object is\n"
         "excluded by identity, not by distance: another object coincident\n"
         "with it is at distance 0 and may be the answer.")

    .def("nearest_objects",
         [](const meshsearch::MeshGrid& self, const unsigned int n,
            const Coords& x, const Coords& y, const Coords& z) {
           const size_t queries = batchSize(x, y, z);
           const double* px = x.data();
           const double* py = y.data();
           const double* pz = z.data();
           std::vector<unsigned int> out(queries*size_t(n));
           {
             nb::gil_scoped_release released;
             for (size_t i=0; i<queries; ++i) {
               try {
                 const auto found = self.nearestObjects(n, px[i], py[i], pz[i]);
                 std::copy(found.begin(), found.end(), out.begin()+i*size_t(n));
               }
               catch (const meshsearch::IndexError& e) { throw meshsearch::IndexError(queryPrefix(i)+e.what()); }
               catch (const meshsearch::Error& e) { throw meshsearch::Error(queryPrefix(i)+e.what()); }
             }
           }
           return indexArray2D(std::move(out), queries, n);
         },
         "n"_a, "x"_a, "y"_a, "z"_a,
         "The n objects nearest to each of an array of points, nearest first:\n"
         "an (n_query, n) array, rectangular because every query returns\n"
         "exactly n.")

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
         "The n objects nearest to an object, nearest first. The query object\n"
         "is excluded by identity, not by distance: another object coincident\n"
         "with it is at distance 0 and is among them.")

    .def("close_objects",
         [](const meshsearch::MeshGrid& self, const Coords& x, const Coords& y, const Coords& z,
            const double rmax, const double rmin) {
           const size_t queries = batchSize(x, y, z);
           const double* px = x.data();
           const double* py = y.data();
           const double* pz = z.data();
           std::vector<unsigned int> indices;
           std::vector<unsigned int> offsets(queries+1, 0);
           {
             nb::gil_scoped_release released;
             for (size_t i=0; i<queries; ++i) {
               try {
                 const auto found = self.closeObjects(px[i], py[i], pz[i], rmax, rmin);
                 indices.insert(indices.end(), found.begin(), found.end());
               }
               catch (const meshsearch::IndexError& e) { throw meshsearch::IndexError(queryPrefix(i)+e.what()); }
               catch (const meshsearch::Error& e) { throw meshsearch::Error(queryPrefix(i)+e.what()); }
               if (indices.size() > std::numeric_limits<unsigned int>::max())
                 throw meshsearch::Error("the batch holds more results than a uint32 offset can address");
               offsets[i+1] = static_cast<unsigned int>(indices.size());
             }
           }
           return nb::make_tuple(indexArray(std::move(indices)), indexArray(std::move(offsets)));
         },
         "x"_a, "y"_a, "z"_a, "rmax"_a, "rmin"_a = 0.,
         "Objects at distance d from each of an array of points, with\n"
         "rmin <= d <= rmax, as (indices, offsets) in compressed-row form:\n"
         "the results of query i are indices[offsets[i]:offsets[i+1]].\n"
         "offsets has n_query + 1 entries, starts at 0 and ends at\n"
         "len(indices). A query with no results gives an empty slice, never a\n"
         "missing entry.")

    .def("close_objects",
         [](const meshsearch::MeshGrid& self, const double x, const double y, const double z,
            const double rmax, const double rmin) {
           return indexArray(self.closeObjects(x, y, z, rmax, rmin));
         },
         "x"_a, "y"_a, "z"_a, "rmax"_a, "rmin"_a = 0.,
         "Objects at distance d from a point, with rmin <= d <= rmax, in\n"
         "unspecified order. The interval is closed at both ends, so an object\n"
         "at exactly either radius is returned. An object lying on the point\n"
         "is at d = 0 and is returned.")

    .def("close_objects",
         [](const meshsearch::MeshGrid& self, const unsigned int index,
            const double rmax, const double rmin) {
           return indexArray(self.closeObjects(index, rmax, rmin));
         },
         "index"_a, "rmax"_a, "rmin"_a = 0.,
         "Objects at distance d from an object, with rmin <= d <= rmax, in\n"
         "unspecified order. The query object is excluded by identity, not by\n"
         "distance: another object coincident with it is at d = 0 and is\n"
         "returned.")

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
