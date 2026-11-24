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
#include <type_traits>
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

  bool isBool (const nb::handle& value)
  {
    const std::string type = Py_TYPE(value.ptr())->tp_name;
    return PyBool_Check(value.ptr()) || type == "numpy.bool_" || type == "numpy.bool";
  }

  bool flag (const nb::handle& value)
  {
    if (!isBool(value)) throw nb::type_error("periodic must be a bool or a sequence of three bools");
    return PyObject_IsTrue(value.ptr()) == 1;
  }

  nb::tuple triple (const std::array<unsigned int, 3>& values)
  {
    return nb::make_tuple(values[0], values[1], values[2]);
  }

  // Query points for the batched calls. A strict 1-D contiguous double array,
  // so that an array argument matches this and a scalar cannot, which is what
  // keeps the overload dispatch unambiguous.
  using Coords = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

  // Object indices for the batched index calls: a 1-D array of any integer
  // type, taken without conversion and range-checked element by element, so
  // that no index is silently wrapped into another.
  using Indices = nb::ndarray<nb::ndim<1>, nb::device::cpu>;

  // A caller's out= array, taken without conversion and checked in outBuffer.
  using OutArray = nb::ndarray<nb::device::cpu>;

  // Hand a vector's buffer to numpy and let a capsule own it.
  template <class T, size_t D>
  nb::ndarray<nb::numpy, T, nb::ndim<D>> owned (std::vector<T>&& values,
                                                const std::array<size_t, D>& shape)
  {
    auto* held = new std::vector<T>(std::move(values));
    nb::capsule owner(held, [](void* p) noexcept {
      delete static_cast<std::vector<T>*>(p);
    });
    return nb::ndarray<nb::numpy, T, nb::ndim<D>>(held->data(), D, shape.data(), owner);
  }

  size_t batchSize (const Coords& x, const Coords& y, const Coords& z)
  {
    if (x.shape(0) != y.shape(0) || x.shape(0) != z.shape(0))
      throw meshsearch::Error("the query coordinate arrays differ in length");
    return x.shape(0);
  }

  std::vector<double> points (const Coords& a)
  {
    return std::vector<double>(a.data(), a.data()+a.shape(0));
  }

  // A batched call fails where the single-point call would, naming the query
  // that did it, and returns nothing partial.
  std::string queryPrefix (const size_t i)
  {
    return "query "+std::to_string(i)+": ";
  }

  template <class T>
  bool readIndex (const Indices& a, const size_t i, long long& value)
  {
    const T element = static_cast<const T*>(a.data())[i*static_cast<size_t>(a.stride(0))];
    if constexpr (std::is_signed_v<T>) {
      value = static_cast<long long>(element);
      return element >= 0 && static_cast<unsigned long long>(element) <= std::numeric_limits<unsigned int>::max();
    }
    else {
      value = static_cast<long long>(std::min<unsigned long long>(element, std::numeric_limits<long long>::max()));
      return static_cast<unsigned long long>(element) <= std::numeric_limits<unsigned int>::max();
    }
  }

  std::vector<unsigned int> indexVector (const Indices& a)
  {
    const auto type = a.dtype();
    const bool sign = type.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Int);
    const bool unsign = type.code == static_cast<uint8_t>(nb::dlpack::dtype_code::UInt);
    if ((!sign && !unsign) || type.lanes != 1 || (type.bits != 8 && type.bits != 16 && type.bits != 32 && type.bits != 64))
      throw nb::type_error("indices must be a 1-D array of integers");

    std::vector<unsigned int> out(a.shape(0));
    for (size_t i=0; i<out.size(); ++i) {
      long long value = 0;
      bool ok = false;
      switch (type.bits) {
      case 8:  ok = sign ? readIndex<int8_t>(a, i, value)  : readIndex<uint8_t>(a, i, value);  break;
      case 16: ok = sign ? readIndex<int16_t>(a, i, value) : readIndex<uint16_t>(a, i, value); break;
      case 32: ok = sign ? readIndex<int32_t>(a, i, value) : readIndex<uint32_t>(a, i, value); break;
      default: ok = sign ? readIndex<int64_t>(a, i, value) : readIndex<uint64_t>(a, i, value); break;
      }
      if (!ok) throw meshsearch::IndexError(queryPrefix(i)+"index "+std::to_string(value)+" is out of range");
      out[i] = static_cast<unsigned int>(value);
    }
    return out;
  }

  // The caller's out= array, which must already be exactly what the call
  // would return: dtype, shape, C-contiguous and writeable. Nothing is
  // converted, because a converted copy would be a different array and the
  // caller's would silently stay unwritten.
  template <class T>
  T* outBuffer (OutArray& array, const nb::handle out, const std::vector<size_t>& shape, const char* dtype)
  {
    std::string expected = "(";
    for (size_t d=0; d<shape.size(); ++d) expected += (d ? ", " : "")+std::to_string(shape[d]);
    expected += shape.size() == 1 ? ",)" : ")";
    const std::string wanted = "out must be a writeable C-contiguous "+std::string(dtype)+" array of shape "+expected;

    if (!nb::try_cast(out, array, false) || array.dtype() != nb::dtype<T>())
      throw nb::type_error(wanted.c_str());
    bool fits = array.ndim() == shape.size();
    for (size_t d=0; fits && d<shape.size(); ++d) fits = array.shape(d) == shape[d];
    int64_t step = 1;
    for (size_t d=shape.size(); fits && d-- > 0; ) {
      if (shape[d] > 1 && array.stride(d) != step) fits = false;
      step *= static_cast<int64_t>(shape[d]);
    }
    if (!fits) throw meshsearch::Error(wanted);
    return static_cast<T*>(array.data());
  }

  // Counts and offsets reach Python as int64, numpy's standard integer, which
  // indexes, slices and repeats without a cast. Each is bounded by a number
  // of results held in memory, so it fits.
  nb::ndarray<nb::numpy, std::int64_t, nb::ndim<1>> int64Array (const std::vector<std::uint64_t>& values)
  {
    const size_t size = values.size();
    std::vector<std::int64_t> converted(size);
    std::transform(values.begin(), values.end(), converted.begin(),
                   [](const std::uint64_t v) { return static_cast<std::int64_t>(v); });
    return owned<std::int64_t, 1>(std::move(converted), {size});
  }

  void copyCounts (const std::vector<std::uint64_t>& values, std::int64_t* dest)
  {
    std::transform(values.begin(), values.end(), dest,
                   [](const std::uint64_t v) { return static_cast<std::int64_t>(v); });
  }

  // Run a C++ batch member. If it raises, the queries are repeated one by one
  // to find the first that raises on its own, and that error is raised,
  // naming the query; the error path alone pays for this. An error no single
  // query reproduces, such as an n the grid refuses on an empty batch, is
  // raised as it is.
  template <class Batch, class Single>
  void runBatch (const size_t queries, Batch&& batch, Single&& single)
  {
    try {
      batch();
    }
    catch (const meshsearch::Error&) {
      for (size_t i=0; i<queries; ++i) {
        try { single(i); }
        catch (const meshsearch::IndexError& e) { throw meshsearch::IndexError(queryPrefix(i)+e.what()); }
        catch (const meshsearch::Error& e) { throw meshsearch::Error(queryPrefix(i)+e.what()); }
      }
      throw;
    }
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
  error_type.attr("__doc__") = "Thrown on invalid input or on a point outside the box.";

  nb::object bases = nb::make_tuple(error_type, nb::borrow(PyExc_IndexError));
  PyObject* index_error = PyErr_NewExceptionWithDoc(
    "meshsearch.IndexError",
    "Thrown on an index that is out of range or names a removed object.",
    bases.ptr(), nullptr);
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
            const std::optional<std::vector<std::vector<double>>>& limits,
            const nb::object& periodic) {
           meshsearch::MeshGrid::Options options;
           if (isBool(periodic)) {
             options.periodic.fill(flag(periodic));
           }
           else {
             if (!nb::isinstance<nb::sequence>(periodic) || nb::isinstance<nb::str>(periodic) || nb::len(periodic) != 3)
               throw nb::type_error("periodic must be a bool or a sequence of three bools");
             for (size_t a=0; a<3; ++a) options.periodic[a] = flag(periodic[nb::int_(a)]);
           }
           if (limits) {
             if (limits->size() != 3 || (*limits)[0].size() != 2 || (*limits)[1].size() != 2 || (*limits)[2].size() != 2)
               throw meshsearch::Error("the limits matrix must be 3x2");
             std::array<std::array<double,2>,3> box;
             for (int a=0; a<3; ++a) box[a] = {(*limits)[a][0], (*limits)[a][1]};
             options.limits = box;
           }
           new (self) meshsearch::MeshGrid(x, y, z, cellsize, options);
         },
         "x"_a, "y"_a, "z"_a, "cellsize"_a, nb::kw_only(),
         "limits"_a = nb::none(), "periodic"_a = false,
         "Build the grid over the points, which must be of equal length.\n\n"
         "limits, keyword-only, is ((xlo,xhi),(ylo,yhi),(zlo,zhi)); when\n"
         "omitted, the data range padded by a margin proportional to it. Each\n"
         "non-periodic side of the box is grown symmetrically to a whole number\n"
         "of cells, and always contains limits: where rounding would leave it\n"
         "inside them, one more cell is added. An axis of zero extent gets one\n"
         "cell.\n\n"
         "periodic, keyword-only, is a bool for all three axes or a triple of\n"
         "bools, one per axis; the default is False. On a periodic axis the box\n"
         "is exactly limits, which are then required, and distances are\n"
         "minimum-image distances: the displacement along the axis is\n"
         "dx = y - q, replaced by dx - L if dx >= L/2 and by dx + L if\n"
         "dx < -L/2, where L = hi - lo (a displacement of exactly +L/2 is taken\n"
         "as -L/2). Each object is seen once, at its nearest image, and any\n"
         "radius is valid, +inf included. Coordinates are not wrapped: a point\n"
         "outside limits is an error. The periodic sides must be integer\n"
         "multiples of the shortest one, within a relative 1e-12. All periodic\n"
         "axes share one cell side, never larger than cellsize; the value\n"
         "actually used is the cellsize property.\n\n"
         "Raises meshsearch.Error for a cellsize that is not finite and\n"
         "positive, or so small that 1/cellsize overflows a double; limits\n"
         "that are not finite or have lo > hi; a coordinate that is not finite\n"
         "or outside limits; a box whose squared diagonal overflows a double\n"
         "(an extent of about 1e154 along an axis); or a grid too large to\n"
         "count; for a periodic axis without limits, with a period below 2**-1000\n"
         "(zero extent included) or too large for its cells to be addressed, or\n"
         "not an integer multiple of the shortest. MemoryError if the grid cannot\n"
         "be allocated. TypeError if periodic is neither a bool nor three bools.")

    .def("__len__", &meshsearch::MeshGrid::get_nObjects,
         "Number of objects in the grid, excluding removed ones.")

    .def("is_alive", &meshsearch::MeshGrid::isAlive, "index"_a,
         "True if index names an object still in the grid.")

    // The array owns its data through a capsule, so it must not be handed out
    // under the reference_internal policy a property would use by default.
    .def_prop_ro("lims",
                 [](const meshsearch::MeshGrid& self) { return limsArray(self.get_lims()); },
                 nb::rv_policy::move,
                 "The box as a 3x2 array: grown to whole cells on a non-periodic axis,\n"
                 "exactly the limits given on a periodic one.")

    .def_prop_ro("n_cells",
                 [](const meshsearch::MeshGrid& self) { return triple(self.get_nCells()); },
                 "Cells per axis.")

    .def_prop_ro("cellsize", &meshsearch::MeshGrid::get_cellsize,
                 "Cell side. The cellsize requested, unless an axis is periodic,\n"
                 "where it can be slightly smaller.")

    .def_prop_ro("periodic",
                 [](const meshsearch::MeshGrid& self) {
                   const auto p = self.get_periodic();
                   return nb::make_tuple(p[0], p[1], p[2]);
                 },
                 "Per axis, whether the box is periodic: a tuple of three bools.")

    .def("nearest_object",
         [](const meshsearch::MeshGrid& self, const Coords& x, const Coords& y, const Coords& z) {
           const size_t queries = batchSize(x, y, z);
           std::vector<unsigned int> found;
           {
             nb::gil_scoped_release released;
             const auto X = points(x), Y = points(y), Z = points(z);
             runBatch(queries,
                      [&] { self.nearestObject(X, Y, Z, found); },
                      [&] (const size_t i) { self.nearestObject(X[i], Y[i], Z[i]); });
           }
           return indexArray(std::move(found));
         },
         "x"_a, "y"_a, "z"_a,
         "The object nearest to each of an array of points: a uint32 array\n"
         "with one index per query, in query order.\n"
         "A large batch is answered in cell order internally, which is faster;\n"
         "the results are always in query order.\n\n"
         "Raises meshsearch.Error, naming the first query that fails, where\n"
         "the single-point call would, and returns nothing partial; also if\n"
         "the arrays differ in length, or if the grid is empty, even for an\n"
         "empty batch.\n\n"
         "The GIL is released while the queries run, so other Python threads\n"
         "may run, and may query the same grid; none may modify it until the\n"
         "call returns.")

    .def("nearest_object",
         [](const meshsearch::MeshGrid& self, const Indices& indices) {
           std::vector<unsigned int> found;
           {
             nb::gil_scoped_release released;
             const auto objects = indexVector(indices);
             runBatch(objects.size(),
                      [&] { self.nearestObject(objects, found); },
                      [&] (const size_t i) { self.nearestObject(objects[i]); });
           }
           return indexArray(std::move(found));
         },
         "indices"_a.noconvert(),
         "The object nearest to each object of an array of indices, each\n"
         "excluded from its own answer: a uint32 array with one index per\n"
         "query, in query order.\n\n"
         "indices is a 1-D array of any integer dtype, such as the uint32\n"
         "arrays the queries return. It is not converted: a value that is not\n"
         "a valid index raises meshsearch.IndexError rather than being\n"
         "wrapped, and an array that is not of integers raises TypeError.\n\n"
         "A large batch is answered in cell order internally, which is faster;\n"
         "the results are always in query order.\n\n"
         "Raises meshsearch.IndexError, naming the first query that fails, on\n"
         "an index that is out of range or removed, and returns nothing\n"
         "partial; meshsearch.Error if fewer than two objects remain, even\n"
         "for an empty array.\n\n"
         "The GIL is released while the queries run, so other Python threads\n"
         "may run, and may query the same grid; none may modify it until the\n"
         "call returns.")

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
            const Coords& x, const Coords& y, const Coords& z, const nb::object& out) -> nb::object {
           const size_t queries = batchSize(x, y, z);
           OutArray target;
           unsigned int* dest = nullptr;
           if (!out.is_none()) dest = outBuffer<unsigned int>(target, out, {queries, size_t(n)}, "uint32");
           std::vector<unsigned int> found;
           {
             nb::gil_scoped_release released;
             const auto X = points(x), Y = points(y), Z = points(z);
             runBatch(queries,
                      [&] { self.nearestObjects(n, X, Y, Z, found); },
                      [&] (const size_t i) { self.nearestObjects(n, X[i], Y[i], Z[i]); });
             if (dest != nullptr) std::copy(found.begin(), found.end(), dest);
           }
           if (dest != nullptr) return out;
           return nb::cast(owned<unsigned int, 2>(std::move(found), {queries, size_t(n)}));
         },
         "n"_a, "x"_a, "y"_a, "z"_a, nb::kw_only(), "out"_a = nb::none(),
         "The n objects nearest to each of an array of points, nearest first;\n"
         "ties by ascending index: a uint32 array of shape (n_query, n),\n"
         "rectangular because every query returns exactly n. A tie at the\n"
         "last place keeps the lower indices.\n"
         "A large batch is answered in cell order internally, which is faster;\n"
         "the results are always in query order.\n\n"
         "out, keyword-only, is an array to write the result into instead of\n"
         "allocating one, and is then returned. It must already be exactly\n"
         "the result: a writeable, C-contiguous uint32 array of shape\n"
         "(n_query, n). Nothing is converted: a wrong dtype, a read-only array\n"
         "or something that is not an array raises TypeError; a wrong shape\n"
         "or a non-contiguous array raises meshsearch.Error. out is checked\n"
         "before any query runs, and is not written to if the call raises.\n\n"
         "Raises meshsearch.Error, naming the first query that fails, where\n"
         "the single-point call would, and returns nothing partial; also if\n"
         "the arrays differ in length, or if n exceeds the number of objects,\n"
         "even for an empty batch, or if n_query * n exceeds what 64-bit sizes\n"
         "can count.\n\n"
         "The GIL is released while the queries run, so other Python threads\n"
         "may run, and may query the same grid; none may modify it until the\n"
         "call returns.")

    .def("nearest_objects",
         [](const meshsearch::MeshGrid& self, const unsigned int n,
            const Indices& indices, const nb::object& out) -> nb::object {
           const size_t queries = indices.shape(0);
           OutArray target;
           unsigned int* dest = nullptr;
           if (!out.is_none()) dest = outBuffer<unsigned int>(target, out, {queries, size_t(n)}, "uint32");
           std::vector<unsigned int> found;
           {
             nb::gil_scoped_release released;
             const auto objects = indexVector(indices);
             runBatch(queries,
                      [&] { self.nearestObjects(n, objects, found); },
                      [&] (const size_t i) { self.nearestObjects(n, objects[i]); });
             if (dest != nullptr) std::copy(found.begin(), found.end(), dest);
           }
           if (dest != nullptr) return out;
           return nb::cast(owned<unsigned int, 2>(std::move(found), {queries, size_t(n)}));
         },
         "n"_a, "indices"_a.noconvert(), nb::kw_only(), "out"_a = nb::none(),
         "The n objects nearest to each object of an array of indices, nearest\n"
         "first, each object excluded from its own answer; ties by ascending\n"
         "index: a uint32 array of shape (len(indices), n).\n\n"
         "indices is a 1-D array of any integer dtype, such as the uint32\n"
         "arrays the queries return. It is not converted: a value that is not\n"
         "a valid index raises meshsearch.IndexError rather than being\n"
         "wrapped, and an array that is not of integers raises TypeError.\n\n"
         "A large batch is answered in cell order internally, which is faster;\n"
         "the results are always in query order.\n\n"
         "out, keyword-only, is an array to write the result into instead of\n"
         "allocating one, and is then returned. It must already be exactly\n"
         "the result: a writeable, C-contiguous uint32 array of shape\n"
         "(len(indices), n). Nothing is converted: a wrong dtype, a read-only array\n"
         "or something that is not an array raises TypeError; a wrong shape\n"
         "or a non-contiguous array raises meshsearch.Error. out is checked\n"
         "before any query runs, and is not written to if the call raises.\n\n"
         "Raises meshsearch.IndexError, naming the first query that fails, on\n"
         "an index that is out of range or removed, and returns nothing\n"
         "partial; meshsearch.Error if n is not below the number of\n"
         "objects, even for an empty array, or if len(indices) * n exceeds\n"
         "what 64-bit sizes can count.\n\n"
         "The GIL is released while the queries run, so other Python threads\n"
         "may run, and may query the same grid; none may modify it until the\n"
         "call returns.")

    .def("nearest_objects",
         [](const meshsearch::MeshGrid& self, const unsigned int n,
            const double x, const double y, const double z) {
           return indexArray(self.nearestObjects(n, x, y, z));
         },
         "n"_a, "x"_a, "y"_a, "z"_a,
         "The n objects nearest to a point, nearest first; ties by ascending\n"
         "index. A tie at the last place keeps the lower indices.")

    .def("nearest_objects",
         [](const meshsearch::MeshGrid& self, const unsigned int n, const unsigned int index) {
           return indexArray(self.nearestObjects(n, index));
         },
         "n"_a, "index"_a,
         "The n objects nearest to an object, nearest first; ties by ascending\n"
         "index. A tie at the last place keeps the lower indices. The query\n"
         "object is excluded by identity, not by distance: another object\n"
         "coincident with it is at distance 0 and is among them.")

    .def("close_objects",
         [](const meshsearch::MeshGrid& self, const Coords& x, const Coords& y, const Coords& z,
            const double rmax, const double rmin) {
           const size_t queries = batchSize(x, y, z);
           meshsearch::MeshGrid::Batch found;
           {
             nb::gil_scoped_release released;
             const auto X = points(x), Y = points(y), Z = points(z);
             runBatch(queries,
                      [&] { self.closeObjects(X, Y, Z, rmax, rmin, found); },
                      [&] (const size_t i) { self.countObjects(X[i], Y[i], Z[i], rmax, rmin); });
           }
           return nb::make_tuple(indexArray(std::move(found.indices)), int64Array(found.offsets));
         },
         "x"_a, "y"_a, "z"_a, "rmax"_a, "rmin"_a = 0.,
         "Objects at distance d from each of an array of points, with\n"
         "rmin <= d <= rmax, as a tuple (indices, offsets) in compressed-row\n"
         "form: the results of query i are indices[offsets[i]:offsets[i+1]],\n"
         "in the order the single call returns them. indices is uint32 and\n"
         "offsets int64; offsets has n_query + 1 entries, starts at 0 and\n"
         "ends at len(indices). A query with no results gives an empty slice,\n"
         "never a missing entry.\n"
         "A large batch is answered in cell order internally, which is faster;\n"
         "the results are always in query order.\n\n"
         "There is no out= argument: the length of indices is not known until\n"
         "the queries have run, so no array could be supplied in advance.\n"
         "count_objects gives the counts alone.\n\n"
         "Raises meshsearch.Error, naming the first query that fails, where\n"
         "the single-point call would, and returns nothing partial; also if\n"
         "the arrays differ in length, or on the radii, even for an empty\n"
         "batch.\n\n"
         "The GIL is released while the queries run, so other Python threads\n"
         "may run, and may query the same grid; none may modify it until the\n"
         "call returns.")

    .def("close_objects",
         [](const meshsearch::MeshGrid& self, const Indices& indices,
            const double rmax, const double rmin) {
           meshsearch::MeshGrid::Batch found;
           {
             nb::gil_scoped_release released;
             const auto objects = indexVector(indices);
             runBatch(objects.size(),
                      [&] { self.closeObjects(objects, rmax, rmin, found); },
                      [&] (const size_t i) { self.countObjects(objects[i], rmax, rmin); });
           }
           return nb::make_tuple(indexArray(std::move(found.indices)), int64Array(found.offsets));
         },
         "indices"_a.noconvert(), "rmax"_a, "rmin"_a = 0.,
         "Objects at distance d from each object of an array of indices, with\n"
         "rmin <= d <= rmax, each object excluded from its own answer, as a\n"
         "tuple (indices, offsets) in compressed-row form: the results of\n"
         "query i are indices[offsets[i]:offsets[i+1]], in the order the\n"
         "single call returns them. indices is uint32 and offsets int64;\n"
         "offsets has n_query + 1 entries, starts at 0 and ends at\n"
         "len(indices). A query with no results gives an empty slice, never a\n"
         "missing entry.\n"
         "\n"
         "indices is a 1-D array of any integer dtype, such as the uint32\n"
         "arrays the queries return. It is not converted: a value that is not\n"
         "a valid index raises meshsearch.IndexError rather than being\n"
         "wrapped, and an array that is not of integers raises TypeError.\n\n"
         "A large batch is answered in cell order internally, which is faster;\n"
         "the results are always in query order.\n\n"
         "There is no out= argument: the length of indices is not known until\n"
         "the queries have run, so no array could be supplied in advance.\n"
         "count_objects gives the counts alone.\n\n"
         "Raises meshsearch.IndexError, naming the first query that fails, on\n"
         "an index that is out of range or removed, and returns nothing\n"
         "partial; meshsearch.Error on the radii, even for an empty\n"
         "array.\n\n"
         "The GIL is released while the queries run, so other Python threads\n"
         "may run, and may query the same grid; none may modify it until the\n"
         "call returns.")

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

    .def("count_objects",
         [](const meshsearch::MeshGrid& self, const Coords& x, const Coords& y, const Coords& z,
            const double rmax, const double rmin, const nb::object& out) -> nb::object {
           const size_t queries = batchSize(x, y, z);
           OutArray target;
           std::int64_t* dest = nullptr;
           if (!out.is_none()) dest = outBuffer<std::int64_t>(target, out, {queries}, "int64");
           std::vector<std::uint64_t> found;
           {
             nb::gil_scoped_release released;
             const auto X = points(x), Y = points(y), Z = points(z);
             runBatch(queries,
                      [&] { self.countObjects(X, Y, Z, rmax, rmin, found); },
                      [&] (const size_t i) { self.countObjects(X[i], Y[i], Z[i], rmax, rmin); });
             if (dest != nullptr) copyCounts(found, dest);
           }
           if (dest != nullptr) return out;
           return nb::cast(int64Array(found));
         },
         "x"_a, "y"_a, "z"_a, "rmax"_a, "rmin"_a = 0., nb::kw_only(), "out"_a = nb::none(),
         "For each of an array of points, the number of objects close_objects\n"
         "would return for it: an int64 array with one count per query, in\n"
         "query order. The indices are never collected, so this is cheaper\n"
         "than close_objects when only the counts are wanted.\n"
         "A large batch is answered in cell order internally, which is faster;\n"
         "the results are always in query order.\n\n"
         "out, keyword-only, is an array to write the result into instead of\n"
         "allocating one, and is then returned. It must already be exactly\n"
         "the result: a writeable, C-contiguous int64 array of shape\n"
         "(n_query,). Nothing is converted: a wrong dtype, a read-only array\n"
         "or something that is not an array raises TypeError; a wrong shape\n"
         "or a non-contiguous array raises meshsearch.Error. out is checked\n"
         "before any query runs, and is not written to if the call raises.\n\n"
         "Raises meshsearch.Error, naming the first query that fails, where\n"
         "the single-point call would, and returns nothing partial; also if\n"
         "the arrays differ in length, or on the radii, even for an empty\n"
         "batch.\n\n"
         "The GIL is released while the queries run, so other Python threads\n"
         "may run, and may query the same grid; none may modify it until the\n"
         "call returns.")

    .def("count_objects",
         [](const meshsearch::MeshGrid& self, const Indices& indices,
            const double rmax, const double rmin, const nb::object& out) -> nb::object {
           const size_t queries = indices.shape(0);
           OutArray target;
           std::int64_t* dest = nullptr;
           if (!out.is_none()) dest = outBuffer<std::int64_t>(target, out, {queries}, "int64");
           std::vector<std::uint64_t> found;
           {
             nb::gil_scoped_release released;
             const auto objects = indexVector(indices);
             runBatch(queries,
                      [&] { self.countObjects(objects, rmax, rmin, found); },
                      [&] (const size_t i) { self.countObjects(objects[i], rmax, rmin); });
             if (dest != nullptr) copyCounts(found, dest);
           }
           if (dest != nullptr) return out;
           return nb::cast(int64Array(found));
         },
         "indices"_a.noconvert(), "rmax"_a, "rmin"_a = 0., nb::kw_only(), "out"_a = nb::none(),
         "For each object of an array of indices, the number of objects\n"
         "close_objects would return for it, the object itself excluded: an\n"
         "int64 array with one count per index, in order.\n\n"
         "indices is a 1-D array of any integer dtype, such as the uint32\n"
         "arrays the queries return. It is not converted: a value that is not\n"
         "a valid index raises meshsearch.IndexError rather than being\n"
         "wrapped, and an array that is not of integers raises TypeError.\n\n"
         "A large batch is answered in cell order internally, which is faster;\n"
         "the results are always in query order.\n\n"
         "out, keyword-only, is an array to write the result into instead of\n"
         "allocating one, and is then returned. It must already be exactly\n"
         "the result: a writeable, C-contiguous int64 array of shape\n"
         "(len(indices),). Nothing is converted: a wrong dtype, a read-only array\n"
         "or something that is not an array raises TypeError; a wrong shape\n"
         "or a non-contiguous array raises meshsearch.Error. out is checked\n"
         "before any query runs, and is not written to if the call raises.\n\n"
         "Raises meshsearch.IndexError, naming the first query that fails, on\n"
         "an index that is out of range or removed, and returns nothing\n"
         "partial; meshsearch.Error on the radii, even for an empty\n"
         "array.\n\n"
         "The GIL is released while the queries run, so other Python threads\n"
         "may run, and may query the same grid; none may modify it until the\n"
         "call returns.")

    .def("count_objects",
         nb::overload_cast<double, double, double, double, double>(&meshsearch::MeshGrid::countObjects, nb::const_),
         "x"_a, "y"_a, "z"_a, "rmax"_a, "rmin"_a = 0.,
         "Number of objects close_objects would return for a point, counted\n"
         "without collecting them.")

    .def("count_objects",
         nb::overload_cast<unsigned int, double, double>(&meshsearch::MeshGrid::countObjects, nb::const_),
         "index"_a, "rmax"_a, "rmin"_a = 0.,
         "Number of objects close_objects would return for an object, which\n"
         "is excluded, counted without collecting them.")

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
