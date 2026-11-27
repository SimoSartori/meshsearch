// nanobind bindings for meshsearch::MeshGrid.
//
// The C++ names are unchanged; Python gets snake_case. Only what
// include/meshsearch/MeshGrid.h declares is exposed. The no-argument getters
// become read-only properties and the get_* spellings are not bound, the live
// object count is reached through len(), and index lists come back as numpy
// uint32 arrays.

#include "meshsearch/MeshGrid.h"
#include "meshsearch/Version.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
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

  LimsArray limsArray (const std::array<std::array<double,2>,3>& lims)
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

  // A caller's out= array, taken without conversion and checked in outBuffer.
  using OutArray = nb::ndarray<nb::device::cpu>;

  // Whatever the buffer protocol hands over, read in place; read-only
  // arrays are as good as writeable ones, since nothing is written to them.
  using Buffer = nb::ndarray<nb::ro, nb::device::cpu>;

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

  // One 0/1 flag per query, as a numpy bool array.
  nb::ndarray<nb::numpy, bool, nb::ndim<1>> boolArray (std::vector<std::uint8_t>&& flags)
  {
    auto* held = new std::vector<std::uint8_t>(std::move(flags));
    nb::capsule owner(held, [](void* p) noexcept {
      delete static_cast<std::vector<std::uint8_t>*>(p);
    });
    const std::array<size_t, 1> shape = {held->size()};
    return nb::ndarray<nb::numpy, bool, nb::ndim<1>>(reinterpret_cast<bool*>(held->data()), 1, shape.data(), owner);
  }

  // An argument whose type has a length is a batch; anything else is a
  // single point or index. The type's slots are read, not called, so a
  // scalar is told apart without raising; only an object whose type has a
  // length it then refuses, a 0-d numpy array, goes through len().
  bool isBatch (const nb::handle& a) noexcept
  {
    PyTypeObject* type = Py_TYPE(a.ptr());
    if (PyType_GetSlot(type, Py_sq_length) == nullptr && PyType_GetSlot(type, Py_mp_length) == nullptr) return false;
    if (PyObject_Length(a.ptr()) >= 0) return true;
    PyErr_Clear();
    return false;
  }

  // The argument as a numpy array, which is the argument itself when it
  // already is one.
  nb::object asArray (const nb::handle& a)
  {
    return nb::module_::import_("numpy").attr("asarray")(a);
  }

  char kindOf (const nb::object& array)
  {
    return nb::cast<std::string>(array.attr("dtype").attr("kind"))[0];
  }

  template <class T>
  T element (const Buffer& a, const size_t i)
  {
    return static_cast<const T*>(a.data())[static_cast<std::int64_t>(i)*a.stride(0)];
  }

  template <class T>
  void widen (const Buffer& a, std::vector<double>& out)
  {
    for (size_t i=0; i<out.size(); ++i) out[i] = static_cast<double>(element<T>(a, i));
  }

  // A 1-D array of query coordinates as doubles. float64 and integer arrays
  // are read where they are, whatever their strides; anything else, a list or
  // a float32 array for instance, is converted by numpy first.
  std::vector<double> readCoords (const nb::handle& argument, const char* name)
  {
    nb::object array = asArray(argument);
    const char kind = kindOf(array);
    if (kind != 'f' && kind != 'i' && kind != 'u')
      throw nb::type_error((std::string(name)+" must be an array of real numbers").c_str());
    if (nb::cast<size_t>(array.attr("ndim")) != 1)
      throw meshsearch::Error(std::string(name)+" must be one-dimensional");

    Buffer a;
    if (kind == 'f' && nb::cast<size_t>(array.attr("itemsize")) != 8) array = array.attr("astype")("float64");
    if (!nb::try_cast(array, a, false))
      throw nb::type_error((std::string(name)+" must be an array of real numbers").c_str());

    std::vector<double> out(a.shape(0));
    const auto type = a.dtype();
    const bool sign = type.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Int);
    switch (type.bits) {
    case 8:  sign ? widen<int8_t>(a, out)  : widen<uint8_t>(a, out);  break;
    case 16: sign ? widen<int16_t>(a, out) : widen<uint16_t>(a, out); break;
    case 32: sign ? widen<int32_t>(a, out) : widen<uint32_t>(a, out); break;
    default:
      if (type.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Float)) widen<double>(a, out);
      else if (sign) widen<int64_t>(a, out);
      else widen<uint64_t>(a, out);
    }
    return out;
  }

  struct Points {
    std::vector<double> x, y, z;
  };

  Points readPoints (const nb::handle& x, const nb::handle& y, const nb::handle& z)
  {
    Points p{readCoords(x, "x"), readCoords(y, "y"), readCoords(z, "z")};
    if (p.x.size() != p.y.size() || p.x.size() != p.z.size())
      throw meshsearch::Error("the query coordinate arrays differ in length");
    return p;
  }

  // Constructor coordinates, which the grid copies during the call. An array
  // already laid out as the C++ constructor reads it, float64 and
  // C-contiguous, is read where it is, through the buffer protocol or
  // DLPack. Any other numpy array of real numbers or bools is converted by
  // numpy in one pass. Anything else, a list or an object array for
  // instance, goes through nanobind's sequence conversion, the one the
  // constructor has always used, so that what it accepts and refuses is
  // unchanged; a shape it cannot take is a meshsearch.Error.
  using Column = nb::ndarray<nb::ro, double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
  using Rows = nb::ndarray<nb::ro, double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

  struct Coords {
    nb::object held;
    const double* in_place = nullptr;
    std::vector<double> converted;
    size_t points = 0;
    const double* data () const { return held.is_valid() ? in_place : converted.data(); }
  };

  // A numpy array whose values numpy itself converts to float64 exactly as
  // the sequence conversion would, element by element.
  bool numpyReal (const nb::handle& a)
  {
    if (!nb::isinstance(a, nb::module_::import_("numpy").attr("ndarray"))) return false;
    const char kind = kindOf(nb::borrow(a));
    return kind == 'f' || kind == 'i' || kind == 'u' || kind == 'b';
  }

  nb::object asFloat64 (const nb::handle& a)
  {
    return nb::module_::import_("numpy").attr("ascontiguousarray")(a, "dtype"_a = "float64");
  }

  // A refused constructor argument is a wrong shape, meshsearch.Error, when
  // numpy reads it as real numbers of another shape, and a TypeError
  // otherwise.
  [[noreturn]] void refuseCoords (const nb::handle& argument, const std::string& wanted, const bool rows)
  {
    try {
      nb::object array = asArray(argument);
      const char kind = kindOf(array);
      if (kind == 'f' || kind == 'i' || kind == 'u' || kind == 'b') {
        const size_t ndim = nb::cast<size_t>(array.attr("ndim"));
        const bool shaped = rows ? ndim == 2 && nb::cast<size_t>(array.attr("shape")[nb::int_(1)]) == 3 : ndim == 1;
        if (!shaped) throw meshsearch::Error(wanted);
      }
    } catch (nb::python_error&) {
    }
    throw nb::type_error(wanted.c_str());
  }

  Coords readColumn (const nb::handle& argument, const char* name)
  {
    const std::string wanted = std::string(name)+" must be a one-dimensional array of real numbers";
    nb::object source = nb::borrow(argument);
    Column column;
    if (numpyReal(source)) {
      if (nb::cast<size_t>(source.attr("ndim")) != 1) throw meshsearch::Error(wanted);
      if (!nb::try_cast(source, column, false)) source = asFloat64(source);
    }
    Coords c;
    if (nb::try_cast(source, column, false)) {
      c.held = source;
      c.in_place = column.data();
      c.points = column.shape(0);
      return c;
    }
    if (!nb::try_cast(source, c.converted, true)) refuseCoords(argument, wanted, false);
    c.points = c.converted.size();
    return c;
  }

  Coords readRows (const nb::handle& argument)
  {
    const char* const wanted = "pos must be an array of real numbers of shape (N, 3)";
    nb::object source = nb::borrow(argument);
    Rows rows;
    if (numpyReal(source)) {
      if (nb::cast<size_t>(source.attr("ndim")) != 2 || nb::cast<size_t>(source.attr("shape")[nb::int_(1)]) != 3)
        throw meshsearch::Error(wanted);
      if (!nb::try_cast(source, rows, false)) source = asFloat64(source);
    }
    Coords c;
    if (nb::try_cast(source, rows, false)) {
      c.held = source;
      c.in_place = rows.data();
      c.points = rows.shape(0);
      return c;
    }
    std::vector<std::array<double, 3>> triples;
    if (!nb::try_cast(source, triples, true)) refuseCoords(argument, wanted, true);
    c.converted.resize(3*triples.size());
    for (size_t i=0; i<triples.size(); ++i)
      for (size_t a=0; a<3; ++a) c.converted[3*i+a] = triples[i][a];
    c.points = triples.size();
    return c;
  }

  // The keywords both constructor forms share.
  meshsearch::MeshGrid::Options readOptions (const std::optional<std::vector<std::vector<double>>>& limits,
                                             const nb::object& periodic)
  {
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
    return options;
  }

  // A batched call fails where the single-point call would, naming the query
  // that did it, and returns nothing partial.
  std::string queryPrefix (const size_t i)
  {
    return "query "+std::to_string(i)+": ";
  }

  template <class T>
  bool readIndex (const Buffer& a, const size_t i, long long& value)
  {
    const T item = element<T>(a, i);
    if constexpr (std::is_signed_v<T>) {
      value = static_cast<long long>(item);
      return item >= 0 && static_cast<unsigned long long>(item) <= std::numeric_limits<unsigned int>::max();
    }
    else {
      value = static_cast<long long>(std::min<unsigned long long>(item, std::numeric_limits<long long>::max()));
      return static_cast<unsigned long long>(item) <= std::numeric_limits<unsigned int>::max();
    }
  }

  std::string outOfRange (const long long value)
  {
    return "index "+std::to_string(value)+" is out of range";
  }

  // Object indices from an integer array-like, taken as they are, whatever
  // the integer type, and range-checked element by element so that no index
  // is wrapped into another. With lenient set, an element that is not a valid
  // index becomes the one value that never names an object, for isAlive,
  // which answers false for it instead of raising.
  std::vector<unsigned int> readIndices (const nb::handle& argument, const bool lenient = false)
  {
    nb::object array = asArray(argument);
    const char kind = kindOf(array);
    const bool empty = nb::cast<size_t>(array.attr("size")) == 0;
    if (kind != 'i' && kind != 'u' && !empty)
      throw nb::type_error("indices must be an array of integers");
    if (nb::cast<size_t>(array.attr("ndim")) != 1)
      throw meshsearch::Error("indices must be one-dimensional");
    if (empty) return {};

    Buffer a;
    if (!nb::try_cast(array, a, false))
      throw nb::type_error("indices must be an array of integers");
    const auto type = a.dtype();
    const bool sign = type.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Int);

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
      if (ok) out[i] = static_cast<unsigned int>(value);
      else if (lenient) out[i] = std::numeric_limits<unsigned int>::max();
      else throw meshsearch::IndexError(queryPrefix(i)+outOfRange(value));
    }
    return out;
  }

  // A single index. A negative one, or one beyond 32 bits, is out of range,
  // not a type error: it raises IndexError, as an index the grid does not
  // hold does. With lenient set it is returned as the value that never names
  // an object.
  unsigned int scalarIndex (const nb::handle& argument, const bool lenient = false)
  {
    nb::object number = nb::steal(PyNumber_Index(argument.ptr()));
    if (!number.is_valid()) throw nb::python_error();
    int overflow = 0;
    const long long value = PyLong_AsLongLongAndOverflow(number.ptr(), &overflow);
    if (value == -1 && PyErr_Occurred()) throw nb::python_error();
    if (overflow == 0 && value >= 0 && value <= static_cast<long long>(std::numeric_limits<unsigned int>::max()))
      return static_cast<unsigned int>(value);
    if (lenient) return std::numeric_limits<unsigned int>::max();
    throw meshsearch::IndexError("index "+nb::cast<std::string>(nb::str(number))+" is out of range");
  }

  // A neighbour count. A negative one is an error of the grid's own kind.
  unsigned int readCount (const nb::handle& argument)
  {
    nb::object number = nb::steal(PyNumber_Index(argument.ptr()));
    if (!number.is_valid()) throw nb::python_error();
    int overflow = 0;
    const long long value = PyLong_AsLongLongAndOverflow(number.ptr(), &overflow);
    if (value == -1 && PyErr_Occurred()) throw nb::python_error();
    if (overflow < 0 || value < 0) throw meshsearch::Error("n must not be negative");
    if (overflow > 0 || value > static_cast<long long>(std::numeric_limits<unsigned int>::max()))
      throw meshsearch::Error("n exceeds the number of objects");
    return static_cast<unsigned int>(value);
  }

  // The workers= keyword of the batch forms: an integer, read without
  // conversion from a float, then checked as meshsearch::Workers checks it.
  meshsearch::Workers readWorkers (const nb::handle& argument)
  {
    nb::object number = nb::steal(PyNumber_Index(argument.ptr()));
    if (!number.is_valid()) throw nb::python_error();
    int overflow = 0;
    const long long value = PyLong_AsLongLongAndOverflow(number.ptr(), &overflow);
    if (value == -1 && PyErr_Occurred()) throw nb::python_error();
    if (overflow != 0) return meshsearch::Workers(overflow < 0 ? -2 : std::numeric_limits<int>::max());
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
      return meshsearch::Workers(value < 0 ? -2 : std::numeric_limits<int>::max());
    return meshsearch::Workers(static_cast<int>(value));
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

  // The named tuple the batched close_objects returns, looked up where it is
  // defined, the module, so that it is the one type a caller can import.
  nb::object makeBatch (IndexArray&& indices, nb::ndarray<nb::numpy, std::int64_t, nb::ndim<1>>&& offsets)
  {
    return nb::module_::import_("meshsearch").attr("Batch")(std::move(indices), std::move(offsets));
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

  // The first argument of a batch form, and the index of a single-object
  // form. Overload resolution matches them on isBatch, so a call reaches the
  // form it is meant for without any overload raising to pass it on.
  struct BatchArg {
    nb::object value;
    operator nb::handle () const { return value; }
  };

  struct ScalarArg {
    nb::object value;
    operator nb::handle () const { return value; }
  };

}

namespace nanobind::detail {

  template <class Arg, bool batch>
  struct ShapeCaster {
    NB_TYPE_CASTER(Arg, const_name("object"))

    bool from_python (handle src, uint8_t, cleanup_list*) noexcept
    {
      if (isBatch(src) != batch) return false;
      value.value = borrow(src);
      return true;
    }

    static handle from_cpp (const Arg& arg, rv_policy, cleanup_list*) noexcept
    {
      return arg.value.inc_ref();
    }
  };

  template <> struct type_caster<BatchArg> : ShapeCaster<BatchArg, true> {};
  template <> struct type_caster<ScalarArg> : ShapeCaster<ScalarArg, false> {};

}


NB_MODULE(meshsearch, m)
{
  m.doc() =
    "A uniform cubic mesh over a set of points, for neighbour queries.\n\n"
    "MeshGrid holds the points and answers the queries; Batch is what a\n"
    "batched close_objects returns; Error and IndexError are the errors the\n"
    "grid raises.";

  // The C++ hierarchy is mirrored: meshsearch::IndexError derives from
  // meshsearch::Error, so meshsearch.IndexError derives from meshsearch.Error
  // as well as from Python's IndexError. Either base catches it, exactly as
  // catch (const Error&) and catch (const IndexError&) both do in C++.
  nb::object error_type = nb::exception<meshsearch::Error>(m, "Error", PyExc_ValueError);
  error_type.attr("__doc__") = "Raised on invalid input: a point outside the box, an invalid argument,\n"
    "or a query the grid cannot answer.";

  nb::object bases = nb::make_tuple(error_type, nb::borrow(PyExc_IndexError));
  PyObject* index_error = PyErr_NewExceptionWithDoc(
    "meshsearch.IndexError",
    "Raised on an index that is negative, out of range or names a removed\n"
    "object. Also a meshsearch.Error and a built-in IndexError.",
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

  m.attr("__version__") = MESHSEARCH_VERSION;

  nb::object batch = nb::module_::import_("collections").attr("namedtuple")(
    "Batch", nb::make_tuple("indices", "offsets"));
  batch.attr("__module__") = "meshsearch";
  batch.attr("__doc__") =
    "What the batched close_objects returns: a named tuple (indices, offsets)\n"
    "in compressed-row form. The results of query i are\n"
    "indices[offsets[i]:offsets[i+1]].";
  batch.attr("indices").attr("__doc__") = "uint32 array: the indices found by every query, concatenated.";
  batch.attr("offsets").attr("__doc__") =
    "int64 array of n_query + 1 entries, starting at 0 and ending at len(indices).";
  m.attr("Batch") = batch;

  nb::class_<meshsearch::MeshGrid>(m, "MeshGrid",
      "Uniform cubic mesh over a set of points, for neighbour queries.\n\n"
      "Objects are identified by their position in the coordinate arrays,\n"
      "or by their row in pos.\n"
      "Indices are stable: remove_object leaves an index permanently invalid,\n"
      "add_object issues a new one, and indices are never reused. Every member\n"
      "taking an index raises meshsearch.IndexError on one that is negative, out\n"
      "of range or removed, except is_alive, which answers False.\n\n"
      "Several threads may query one grid at the same time; add_object and\n"
      "remove_object require that no other thread use it.\n\n"
      "Batched queries take any array-like for the coordinates: a float64 or\n"
      "integer array is used as it is, and anything else, a list or a float32\n"
      "array for instance, is converted by numpy first.\n\n"
      "A grid can be copied with copy.copy and copy.deepcopy; the copy is\n"
      "independent and answers every query as the original does. Grids are not\n"
      "picklable: for parallel work use threads, which share one grid, or\n"
      "rebuild the grid in each process from the coordinates, which is fast.")

    .def("__init__",
         [](meshsearch::MeshGrid* self,
            const nb::handle& x,
            const nb::handle& y,
            const nb::handle& z,
            const double cellsize,
            const std::optional<std::vector<std::vector<double>>>& limits,
            const nb::object& periodic) {
           const Coords X = readColumn(x, "x"), Y = readColumn(y, "y"), Z = readColumn(z, "z");
           const meshsearch::MeshGrid::Options options = readOptions(limits, periodic);
           if (X.points != Y.points || X.points != Z.points)
             throw meshsearch::Error("the coordinate vectors differ in length");
           new (self) meshsearch::MeshGrid(X.data(), Y.data(), Z.data(), X.points, cellsize, options);
         },
         "x"_a, "y"_a, "z"_a, "cellsize"_a, nb::kw_only(),
         "limits"_a = nb::none(), "periodic"_a = false,
         "Build the grid over the points x, y, z, one-dimensional arrays of\n"
         "equal length; object i is the point (x[i], y[i], z[i]). The grid\n"
         "copies the coordinates and keeps no reference to the arrays. A\n"
         "float64 C-contiguous array is read where it is; any other\n"
         "array-like of real numbers, a float32 array, a strided column or a\n"
         "list for instance, is converted first, which costs a temporary copy.\n"
         "For points held as rows, use MeshGrid(pos, cellsize) instead.\n\n"
         "Coordinates, cellsize and every radius\n"
         "must be in the same unit. cellsize is the cell side, the one tuning\n"
         "parameter: for uniform data, start between 1.5 and 2.5 mean\n"
         "separations, the mean separation being (volume / number of points)\n"
         "** (1/3); the guide discusses other choices.\n\n"
         "limits, keyword-only, is the box ((xlo,xhi),(ylo,yhi),(zlo,zhi)), and\n"
         "must contain every point. When omitted, the box is fitted to the\n"
         "points with a small margin, and a query point outside their range\n"
         "may raise meshsearch.Error: give limits whenever queries can fall\n"
         "outside the data. Each non-periodic side of the box is grown\n"
         "symmetrically to a whole number of cells and always contains the\n"
         "limits; an axis of zero extent gets one cell.\n\n"
         "periodic, keyword-only, is a bool for all three axes or a triple of\n"
         "bools; the default is False. A periodic axis requires limits, and its\n"
         "box is exactly them, with period L = hi - lo. Distances along it are\n"
         "minimum-image distances, the displacement taken in [-L/2, L/2), so\n"
         "each object is seen once and any radius is valid, +inf included.\n"
         "Coordinates are not wrapped. The periods must be integer multiples\n"
         "of the shortest one, within a relative 1e-12. All cells share one\n"
         "side, never larger than cellsize; the cellsize property gives it.\n\n"
         "Raises meshsearch.Error if the arrays differ in length or hold more\n"
         "than 4294967295 points; if a coordinate is not finite or lies outside\n"
         "limits; if cellsize is not finite and positive, or so small that\n"
         "1/cellsize overflows; if limits is not 3x2, not finite, or has\n"
         "lo > hi; if there are no points and no limits; if the box is so large\n"
         "that its squared diagonal overflows, or the grid would need more than\n"
         "2147483647 cells along an axis; and, for a periodic axis, if limits\n"
         "is missing, the period is below 2**-1000 or the limits too large to\n"
         "shift by it, or the period is not an integer multiple of the\n"
         "shortest. TypeError if periodic is neither a bool nor three bools,\n"
         "or the coordinates are not arrays of real numbers;\n"
         "meshsearch.Error if they are not one-dimensional.\n"
         "MemoryError if the grid cannot be allocated.")

    .def("__init__",
         [](meshsearch::MeshGrid* self,
            const nb::handle& pos,
            const double cellsize,
            const std::optional<std::vector<std::vector<double>>>& limits,
            const nb::object& periodic) {
           const Coords P = readRows(pos);
           const meshsearch::MeshGrid::Options options = readOptions(limits, periodic);
           new (self) meshsearch::MeshGrid(P.data(), P.points, cellsize, options);
         },
         "pos"_a, "cellsize"_a, nb::kw_only(),
         "limits"_a = nb::none(), "periodic"_a = false,
         "Build the grid over the points pos, an array of shape (N, 3): object\n"
         "i is the point pos[i] = (x, y, z). The grid copies the coordinates\n"
         "and keeps no reference to the array, and is the grid\n"
         "MeshGrid(pos[:, 0], pos[:, 1], pos[:, 2], cellsize) builds, with the\n"
         "same results in the same order. A float64 C-contiguous array, the\n"
         "usual layout of an (N, 3) numpy array, is read where it is; any\n"
         "other array-like of real numbers of that shape, a float32 or a\n"
         "Fortran-ordered array or a list of triples for instance, is\n"
         "converted first, which costs a temporary copy.\n\n"
         "cellsize, limits and periodic are those of MeshGrid(x, y, z,\n"
         "cellsize), and so are the errors, except that the coordinates cannot\n"
         "differ in length: TypeError if pos is not an array of real numbers,\n"
         "meshsearch.Error if it is not of shape (N, 3).")

    .def("__len__", &meshsearch::MeshGrid::get_nObjects,
         "Number of objects in the grid, excluding removed ones.")

    .def("is_alive",
         [](const meshsearch::MeshGrid& self, const ScalarArg& index) {
           return self.isAlive(scalarIndex(index, true));
         },
         "index"_a,
         "True if index names an object still in the grid. A negative index,\n"
         "or one out of range or removed, is False: the one member taking an\n"
         "index that does not raise IndexError.\n\n"
         "Raises TypeError if index is not an integer.")

    .def("is_alive",
         [](const meshsearch::MeshGrid& self, const BatchArg& indices, const nb::object& out) -> nb::object {
           const auto objects = readIndices(indices, true);
           OutArray target;
           std::uint8_t* dest = nullptr;
           if (!out.is_none()) dest = reinterpret_cast<std::uint8_t*>(outBuffer<bool>(target, out, {objects.size()}, "bool"));
           std::vector<std::uint8_t> found;
           {
             nb::gil_scoped_release released;
             self.isAlive(objects, found);
             if (dest != nullptr) std::copy(found.begin(), found.end(), dest);
           }
           if (dest != nullptr) return out;
           return nb::cast(boolArray(std::move(found)));
         },
         "indices"_a, nb::kw_only(), "out"_a = nb::none(),
         "For each index of an array-like of integers, whether it names an\n"
         "object still in the grid: a bool array, in order. As for a single\n"
         "index, one that is negative or out of range is False, not an error.\n"
         "An array of integers of any dtype is used as it is; a list or tuple\n"
         "is converted first. Anything that is not integers raises TypeError;\n"
         "anything that is not one-dimensional raises meshsearch.Error.\n\n"
         "out, keyword-only, is an array to write the result into instead of\n"
         "allocating one, and is then returned. It must already be exactly\n"
         "the result: a writeable, C-contiguous bool array of shape\n"
         "(len(indices),). Nothing is converted: a wrong dtype, a read-only array\n"
         "or something that is not an array raises TypeError; a wrong shape\n"
         "or a non-contiguous array raises meshsearch.Error.")

    .def("contains",
         [](const meshsearch::MeshGrid& self, const double x, const double y, const double z) {
           return self.contains(x, y, z);
         },
         "x"_a, "y"_a, "z"_a,
         "True if a query at the point is accepted, that is, if the point lies\n"
         "in the box. False for a NaN coordinate. Raises nothing for any real\n"
         "coordinates; TypeError if one is not a real number.")

    .def("contains",
         [](const meshsearch::MeshGrid& self, const BatchArg& x, const nb::object& y, const nb::object& z,
            const nb::object& out) -> nb::object {
           const Points points = readPoints(x, y, z);
           const std::vector<double>& X = points.x; const std::vector<double>& Y = points.y; const std::vector<double>& Z = points.z;
           OutArray target;
           std::uint8_t* dest = nullptr;
           if (!out.is_none()) dest = reinterpret_cast<std::uint8_t*>(outBuffer<bool>(target, out, {X.size()}, "bool"));
           std::vector<std::uint8_t> found;
           {
             nb::gil_scoped_release released;
             self.contains(X, Y, Z, found);
             if (dest != nullptr) std::copy(found.begin(), found.end(), dest);
           }
           if (dest != nullptr) return out;
           return nb::cast(boolArray(std::move(found)));
         },
         "x"_a, "y"_a, "z"_a, nb::kw_only(), "out"_a = nb::none(),
         "For each of an array of points, whether a query at it is accepted,\n"
         "that is, whether it lies in the box: a bool array, in query order.\n"
         "False for a NaN coordinate; no point makes it raise.\n\n"
         "out, keyword-only, is an array to write the result into instead of\n"
         "allocating one, and is then returned. It must already be exactly\n"
         "the result: a writeable, C-contiguous bool array of shape\n"
         "(n_query,). Nothing is converted: a wrong dtype, a read-only array\n"
         "or something that is not an array raises TypeError; a wrong shape\n"
         "or a non-contiguous array raises meshsearch.Error.\n\n"
         "Raises meshsearch.Error if the arrays differ in length or are not\n"
         "one-dimensional; TypeError if they do not hold real numbers.")

    .def("__repr__",
         [](const meshsearch::MeshGrid& self) {
           return nb::str("MeshGrid(objects={}, n_cells={}, cellsize={!r}, periodic={})").attr("format")(
             self.get_nObjects(), triple(self.get_nCells()), self.get_cellsize(),
             nb::make_tuple(self.get_periodic()[0], self.get_periodic()[1], self.get_periodic()[2]));
         },
         "A short description: the object count, cells per axis, cell side and\n"
         "periodic axes.")

    .def("__copy__", [](const meshsearch::MeshGrid& self) { return meshsearch::MeshGrid(self); },
         "A copy of the grid, independent of it and answering every query as it does.")

    .def("__deepcopy__", [](const meshsearch::MeshGrid& self, const nb::object&) { return meshsearch::MeshGrid(self); },
         "memo"_a,
         "A copy of the grid, independent of it and answering every query as it does.")
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
                 "where it is the largest side not above the request that fits a\n"
                 "whole number of times into the shortest period.")

    .def_prop_ro("periodic",
                 [](const meshsearch::MeshGrid& self) {
                   const auto p = self.get_periodic();
                   return nb::make_tuple(p[0], p[1], p[2]);
                 },
                 "Per axis, whether the box is periodic: a tuple of three bools.")

    .def("nearest_object",
         nb::overload_cast<double, double, double>(&meshsearch::MeshGrid::nearestObject, nb::const_),
         "x"_a, "y"_a, "z"_a,
         "Index of the object nearest to a point. An object lying on the\n"
         "point is at distance 0 and is the answer; ties go to the lower index.\n\n"
         "Raises meshsearch.Error if the point is outside the box or the grid\n"
         "holds no objects.")

    .def("nearest_object",
         [](const meshsearch::MeshGrid& self, const ScalarArg& index) {
           return self.nearestObject(scalarIndex(index));
         },
         "index"_a,
         "Index of the object nearest to an object. The query object is\n"
         "excluded by identity, not by distance: another object coincident\n"
         "with it is at distance 0 and may be the answer.\n\n"
         "Raises meshsearch.IndexError if index is negative, out of range or\n"
         "removed; meshsearch.Error if no other object remains; TypeError if\n"
         "index is not an integer.")

    .def("nearest_object",
         [](const meshsearch::MeshGrid& self, const BatchArg& x, const nb::object& y, const nb::object& z,
            const nb::object& out, const nb::object& workers) -> nb::object {
           const meshsearch::Workers threads = readWorkers(workers);
           const Points points = readPoints(x, y, z);
           const std::vector<double>& X = points.x; const std::vector<double>& Y = points.y; const std::vector<double>& Z = points.z;
           const size_t queries = X.size();
           OutArray target;
           unsigned int* dest = nullptr;
           if (!out.is_none()) dest = outBuffer<unsigned int>(target, out, {queries}, "uint32");
           std::vector<unsigned int> found;
           {
             nb::gil_scoped_release released;
             runBatch(queries,
                      [&] { self.nearestObject(X, Y, Z, found, threads); },
                      [&] (const size_t i) { self.nearestObject(X[i], Y[i], Z[i]); });
             if (dest != nullptr) std::copy(found.begin(), found.end(), dest);
           }
           if (dest != nullptr) return out;
           return nb::cast(indexArray(std::move(found)));
         },
         "x"_a, "y"_a, "z"_a, nb::kw_only(), "out"_a = nb::none(), "workers"_a = 1,
         "The object nearest to each of an array-like of points: a uint32 array\n"
         "with one index per query, in query order.\n"
         "A large batch is answered in cell order internally, which is faster;\n"
         "the results are always in query order.\n\n"
         "out, keyword-only, is an array to write the result into instead of\n"
         "allocating one, and is then returned. It must already be exactly\n"
         "the result: a writeable, C-contiguous uint32 array of shape\n"
         "(n_query,). Nothing is converted: a wrong dtype, a read-only array\n"
         "or something that is not an array raises TypeError; a wrong shape\n"
         "or a non-contiguous array raises meshsearch.Error. out is checked\n"
         "before any query runs, and is not written to if the call raises.\n\n"
         "Raises meshsearch.Error, naming the first query that fails, where\n"
         "the single-point call would, and returns nothing partial; also if\n"
         "the arrays differ in length, or if the grid is empty, even for an\n"
         "empty batch.\n\n"
         "The GIL is released while the queries run, so other Python threads\n"
         "may run, and may query the same grid; none may modify it until the\n"
         "call returns.\n\n"
         "workers, keyword-only, is the number of threads the batch may use:\n"
         "1, the default, runs it in the calling thread; -1 uses every core\n"
         "available to the process (on Linux, its CPU affinity set). The\n"
         "results and their order do not depend on it. A batch too small to\n"
         "share uses fewer threads. Raises meshsearch.Error if workers is 0 or\n"
         "below -1, TypeError if it is not an integer.")

    .def("nearest_object",
         [](const meshsearch::MeshGrid& self, const BatchArg& indices, const nb::object& out,
            const nb::object& workers) -> nb::object {
           const meshsearch::Workers threads = readWorkers(workers);
           const auto objects = readIndices(indices);
           const size_t queries = objects.size();
           OutArray target;
           unsigned int* dest = nullptr;
           if (!out.is_none()) dest = outBuffer<unsigned int>(target, out, {queries}, "uint32");
           std::vector<unsigned int> found;
           {
             nb::gil_scoped_release released;
             runBatch(objects.size(),
                      [&] { self.nearestObject(objects, found, threads); },
                      [&] (const size_t i) { self.nearestObject(objects[i]); });
             if (dest != nullptr) std::copy(found.begin(), found.end(), dest);
           }
           if (dest != nullptr) return out;
           return nb::cast(indexArray(std::move(found)));
         },
         "indices"_a, nb::kw_only(), "out"_a = nb::none(), "workers"_a = 1,
         "The object nearest to each object of an array of indices, each\n"
         "excluded from its own answer: a uint32 array with one index per\n"
         "query, in query order.\n\n"
         "indices is a 1-D array-like of integers, such as the uint32 arrays the\n"
         "queries return. An array of any integer dtype is read as it is,\n"
         "never cast, so values are never wrapped: a negative value or\n"
         "one that is not a valid index raises meshsearch.IndexError. A list or\n"
         "tuple of integers is converted first. Anything that is not integers\n"
         "raises TypeError; anything that is not one-dimensional raises\n"
         "meshsearch.Error.\n\n"
         "A large batch is answered in cell order internally, which is faster;\n"
         "the results are always in query order.\n\n"
         "out, keyword-only, is an array to write the result into instead of\n"
         "allocating one, and is then returned. It must already be exactly\n"
         "the result: a writeable, C-contiguous uint32 array of shape\n"
         "(len(indices),). Nothing is converted: a wrong dtype, a read-only array\n"
         "or something that is not an array raises TypeError; a wrong shape\n"
         "or a non-contiguous array raises meshsearch.Error. out is checked\n"
         "before any query runs, and is not written to if the call raises.\n\n"
         "Raises meshsearch.IndexError, naming the first query that fails, on\n"
         "an index that is negative, out of range or removed, and returns nothing\n"
         "partial; meshsearch.Error if fewer than two objects remain, even\n"
         "for an empty array.\n\n"
         "The GIL is released while the queries run, so other Python threads\n"
         "may run, and may query the same grid; none may modify it until the\n"
         "call returns.\n\n"
         "workers, keyword-only, is the number of threads the batch may use:\n"
         "1, the default, runs it in the calling thread; -1 uses every core\n"
         "available to the process (on Linux, its CPU affinity set). The\n"
         "results and their order do not depend on it. A batch too small to\n"
         "share uses fewer threads. Raises meshsearch.Error if workers is 0 or\n"
         "below -1, TypeError if it is not an integer.")

    .def("nearest_objects",
         [](const meshsearch::MeshGrid& self, const nb::object& n,
            const double x, const double y, const double z) {
           return indexArray(self.nearestObjects(readCount(n), x, y, z));
         },
         "n"_a, "x"_a, "y"_a, "z"_a,
         "The n objects nearest to a point, nearest first, as a uint32 array;\n"
         "ties by ascending index, so a tie at the last place keeps the lower\n"
         "indices. n = 0 returns an empty array.\n\n"
         "Raises meshsearch.Error if the point is outside the box, or if n is\n"
         "negative or exceeds the number of objects.")

    .def("nearest_objects",
         [](const meshsearch::MeshGrid& self, const nb::object& n, const ScalarArg& index) {
           const unsigned int count = readCount(n);
           return indexArray(self.nearestObjects(count, scalarIndex(index)));
         },
         "n"_a, "index"_a,
         "The n objects nearest to an object, nearest first; ties by ascending\n"
         "index. A tie at the last place keeps the lower indices. The query\n"
         "object is excluded by identity, not by distance: another object\n"
         "coincident with it is at distance 0 and is among them.\n\n"
         "Raises meshsearch.IndexError if index is negative, out of range or\n"
         "removed; meshsearch.Error if n is negative or not below the number of\n"
         "objects; TypeError if index is not an integer.")

    .def("nearest_objects",
         [](const meshsearch::MeshGrid& self, const nb::object& count,
            const BatchArg& x, const nb::object& y, const nb::object& z, const nb::object& out,
            const nb::object& workers) -> nb::object {
           const unsigned int n = readCount(count);
           const meshsearch::Workers threads = readWorkers(workers);
           const Points points = readPoints(x, y, z);
           const std::vector<double>& X = points.x; const std::vector<double>& Y = points.y; const std::vector<double>& Z = points.z;
           const size_t queries = X.size();
           OutArray target;
           unsigned int* dest = nullptr;
           if (!out.is_none()) dest = outBuffer<unsigned int>(target, out, {queries, size_t(n)}, "uint32");
           std::vector<unsigned int> found;
           {
             nb::gil_scoped_release released;
             runBatch(queries,
                      [&] { self.nearestObjects(n, X, Y, Z, found, threads); },
                      [&] (const size_t i) { self.nearestObjects(n, X[i], Y[i], Z[i]); });
             if (dest != nullptr) std::copy(found.begin(), found.end(), dest);
           }
           if (dest != nullptr) return out;
           return nb::cast(owned<unsigned int, 2>(std::move(found), {queries, size_t(n)}));
         },
         "n"_a, "x"_a, "y"_a, "z"_a, nb::kw_only(), "out"_a = nb::none(), "workers"_a = 1,
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
         "the arrays differ in length, or if n is negative or exceeds the\n"
         "number of objects,\n"
         "even for an empty batch, or if n_query * n exceeds what 64-bit sizes\n"
         "can count.\n\n"
         "The GIL is released while the queries run, so other Python threads\n"
         "may run, and may query the same grid; none may modify it until the\n"
         "call returns.\n\n"
         "workers, keyword-only, is the number of threads the batch may use:\n"
         "1, the default, runs it in the calling thread; -1 uses every core\n"
         "available to the process (on Linux, its CPU affinity set). The\n"
         "results and their order do not depend on it. A batch too small to\n"
         "share uses fewer threads. Raises meshsearch.Error if workers is 0 or\n"
         "below -1, TypeError if it is not an integer.")

    .def("nearest_objects",
         [](const meshsearch::MeshGrid& self, const nb::object& count,
            const BatchArg& indices, const nb::object& out, const nb::object& workers) -> nb::object {
           const unsigned int n = readCount(count);
           const meshsearch::Workers threads = readWorkers(workers);
           const auto objects = readIndices(indices);
           const size_t queries = objects.size();
           OutArray target;
           unsigned int* dest = nullptr;
           if (!out.is_none()) dest = outBuffer<unsigned int>(target, out, {queries, size_t(n)}, "uint32");
           std::vector<unsigned int> found;
           {
             nb::gil_scoped_release released;
             runBatch(queries,
                      [&] { self.nearestObjects(n, objects, found, threads); },
                      [&] (const size_t i) { self.nearestObjects(n, objects[i]); });
             if (dest != nullptr) std::copy(found.begin(), found.end(), dest);
           }
           if (dest != nullptr) return out;
           return nb::cast(owned<unsigned int, 2>(std::move(found), {queries, size_t(n)}));
         },
         "n"_a, "indices"_a, nb::kw_only(), "out"_a = nb::none(), "workers"_a = 1,
         "The n objects nearest to each object of an array of indices, nearest\n"
         "first, each object excluded from its own answer; ties by ascending\n"
         "index: a uint32 array of shape (len(indices), n).\n\n"
         "indices is a 1-D array-like of integers, such as the uint32 arrays the\n"
         "queries return. An array of any integer dtype is read as it is,\n"
         "never cast, so values are never wrapped: a negative value or\n"
         "one that is not a valid index raises meshsearch.IndexError. A list or\n"
         "tuple of integers is converted first. Anything that is not integers\n"
         "raises TypeError; anything that is not one-dimensional raises\n"
         "meshsearch.Error.\n\n"
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
         "an index that is negative, out of range or removed, and returns nothing\n"
         "partial; meshsearch.Error if n is negative or not below the number of\n"
         "objects, even for an empty array, or if len(indices) * n exceeds\n"
         "what 64-bit sizes can count.\n\n"
         "The GIL is released while the queries run, so other Python threads\n"
         "may run, and may query the same grid; none may modify it until the\n"
         "call returns.\n\n"
         "workers, keyword-only, is the number of threads the batch may use:\n"
         "1, the default, runs it in the calling thread; -1 uses every core\n"
         "available to the process (on Linux, its CPU affinity set). The\n"
         "results and their order do not depend on it. A batch too small to\n"
         "share uses fewer threads. Raises meshsearch.Error if workers is 0 or\n"
         "below -1, TypeError if it is not an integer.")

    .def("close_objects",
         [](const meshsearch::MeshGrid& self, const double x, const double y, const double z,
            const double rmax, const double rmin) {
           return indexArray(self.closeObjects(x, y, z, rmax, rmin));
         },
         "x"_a, "y"_a, "z"_a, "rmax"_a, "rmin"_a = 0.,
         "Objects at distance d from a point, with rmin <= d <= rmax, as a\n"
         "uint32 array whose order is not meaningful. The interval is closed at\n"
         "both ends, so an object at exactly either radius is returned, and\n"
         "rmin == rmax selects the objects at exactly that distance. rmax may be\n"
         "+inf. A ball reaching past the box is cut off at its faces, except\n"
         "along a periodic axis, where it wraps. An object lying on the point\n"
         "is at d = 0 and is returned.\n\n"
         "Raises meshsearch.Error if the point is outside the box, if a radius\n"
         "is negative or NaN, or if rmin exceeds rmax.")

    .def("close_objects",
         [](const meshsearch::MeshGrid& self, const ScalarArg& index,
            const double rmax, const double rmin) {
           return indexArray(self.closeObjects(scalarIndex(index), rmax, rmin));
         },
         "index"_a, "rmax"_a, "rmin"_a = 0.,
         "Objects at distance d from an object, with rmin <= d <= rmax, as a\n"
         "uint32 array whose order is not meaningful; the interval is the\n"
         "point form's. The query object is excluded by identity, not by\n"
         "distance: another object coincident with it is at d = 0 and is\n"
         "returned.\n\n"
         "Raises meshsearch.IndexError if index is negative, out of range or\n"
         "removed; meshsearch.Error if a radius is negative or NaN, or if rmin\n"
         "exceeds rmax; TypeError if index is not an integer.")

    .def("close_objects",
         [](const meshsearch::MeshGrid& self, const BatchArg& x, const nb::object& y, const nb::object& z,
            const double rmax, const double rmin, const nb::object& workers) {
           const meshsearch::Workers threads = readWorkers(workers);
           const Points points = readPoints(x, y, z);
           const std::vector<double>& X = points.x; const std::vector<double>& Y = points.y; const std::vector<double>& Z = points.z;
           const size_t queries = X.size();
           meshsearch::MeshGrid::Batch found;
           {
             nb::gil_scoped_release released;
             runBatch(queries,
                      [&] { self.closeObjects(X, Y, Z, rmax, rmin, found, threads); },
                      [&] (const size_t i) { self.countObjects(X[i], Y[i], Z[i], rmax, rmin); });
           }
           return makeBatch(indexArray(std::move(found.indices)), int64Array(found.offsets));
         },
         "x"_a, "y"_a, "z"_a, "rmax"_a, "rmin"_a = 0., nb::kw_only(), "workers"_a = 1,
         "Objects at distance d from each of an array of points, with\n"
         "rmin <= d <= rmax, as a meshsearch.Batch (indices, offsets) in compressed-row\n"
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
         "call returns.\n\n"
         "workers, keyword-only, is the number of threads the batch may use:\n"
         "1, the default, runs it in the calling thread; -1 uses every core\n"
         "available to the process (on Linux, its CPU affinity set). The\n"
         "results and their order do not depend on it. A batch too small to\n"
         "share uses fewer threads. Raises meshsearch.Error if workers is 0 or\n"
         "below -1, TypeError if it is not an integer.")

    .def("close_objects",
         [](const meshsearch::MeshGrid& self, const BatchArg& indices,
            const double rmax, const double rmin, const nb::object& workers) {
           const meshsearch::Workers threads = readWorkers(workers);
           const auto objects = readIndices(indices);
           meshsearch::MeshGrid::Batch found;
           {
             nb::gil_scoped_release released;
             runBatch(objects.size(),
                      [&] { self.closeObjects(objects, rmax, rmin, found, threads); },
                      [&] (const size_t i) { self.countObjects(objects[i], rmax, rmin); });
           }
           return makeBatch(indexArray(std::move(found.indices)), int64Array(found.offsets));
         },
         "indices"_a, "rmax"_a, "rmin"_a = 0., nb::kw_only(), "workers"_a = 1,
         "Objects at distance d from each object of an array of indices, with\n"
         "rmin <= d <= rmax, each object excluded from its own answer, as a\n"
         "meshsearch.Batch (indices, offsets) in compressed-row form: the results of\n"
         "query i are indices[offsets[i]:offsets[i+1]], in the order the\n"
         "single call returns them. indices is uint32 and offsets int64;\n"
         "offsets has n_query + 1 entries, starts at 0 and ends at\n"
         "len(indices). A query with no results gives an empty slice, never a\n"
         "missing entry.\n"
         "\n"
         "indices is a 1-D array-like of integers, such as the uint32 arrays the\n"
         "queries return. An array of any integer dtype is read as it is,\n"
         "never cast, so values are never wrapped: a negative value or\n"
         "one that is not a valid index raises meshsearch.IndexError. A list or\n"
         "tuple of integers is converted first. Anything that is not integers\n"
         "raises TypeError; anything that is not one-dimensional raises\n"
         "meshsearch.Error.\n\n"
         "A large batch is answered in cell order internally, which is faster;\n"
         "the results are always in query order.\n\n"
         "There is no out= argument: the length of indices is not known until\n"
         "the queries have run, so no array could be supplied in advance.\n"
         "count_objects gives the counts alone.\n\n"
         "Raises meshsearch.IndexError, naming the first query that fails, on\n"
         "an index that is negative, out of range or removed, and returns nothing\n"
         "partial; meshsearch.Error on the radii, even for an empty\n"
         "array.\n\n"
         "The GIL is released while the queries run, so other Python threads\n"
         "may run, and may query the same grid; none may modify it until the\n"
         "call returns.\n\n"
         "workers, keyword-only, is the number of threads the batch may use:\n"
         "1, the default, runs it in the calling thread; -1 uses every core\n"
         "available to the process (on Linux, its CPU affinity set). The\n"
         "results and their order do not depend on it. A batch too small to\n"
         "share uses fewer threads. Raises meshsearch.Error if workers is 0 or\n"
         "below -1, TypeError if it is not an integer.")

    .def("count_objects",
         nb::overload_cast<double, double, double, double, double>(&meshsearch::MeshGrid::countObjects, nb::const_),
         "x"_a, "y"_a, "z"_a, "rmax"_a, "rmin"_a = 0.,
         "Number of objects close_objects would return for a point, counted\n"
         "without collecting them, as an int.\n\n"
         "Raises meshsearch.Error as close_objects does: if the point is\n"
         "outside the box, if a radius is negative or NaN, or if rmin exceeds\n"
         "rmax.")

    .def("count_objects",
         [](const meshsearch::MeshGrid& self, const ScalarArg& index, const double rmax, const double rmin) {
           return self.countObjects(scalarIndex(index), rmax, rmin);
         },
         "index"_a, "rmax"_a, "rmin"_a = 0.,
         "Number of objects close_objects would return for an object, which\n"
         "is excluded, counted without collecting them, as an int.\n\n"
         "Raises meshsearch.IndexError if index is negative, out of range or\n"
         "removed; meshsearch.Error on the radii, as close_objects; TypeError\n"
         "if index is not an integer.")

    .def("count_objects",
         [](const meshsearch::MeshGrid& self, const BatchArg& x, const nb::object& y, const nb::object& z,
            const double rmax, const double rmin, const nb::object& out, const nb::object& workers) -> nb::object {
           const meshsearch::Workers threads = readWorkers(workers);
           const Points points = readPoints(x, y, z);
           const std::vector<double>& X = points.x; const std::vector<double>& Y = points.y; const std::vector<double>& Z = points.z;
           const size_t queries = X.size();
           OutArray target;
           std::int64_t* dest = nullptr;
           if (!out.is_none()) dest = outBuffer<std::int64_t>(target, out, {queries}, "int64");
           std::vector<std::uint64_t> found;
           {
             nb::gil_scoped_release released;
             runBatch(queries,
                      [&] { self.countObjects(X, Y, Z, rmax, rmin, found, threads); },
                      [&] (const size_t i) { self.countObjects(X[i], Y[i], Z[i], rmax, rmin); });
             if (dest != nullptr) copyCounts(found, dest);
           }
           if (dest != nullptr) return out;
           return nb::cast(int64Array(found));
         },
         "x"_a, "y"_a, "z"_a, "rmax"_a, "rmin"_a = 0., nb::kw_only(), "out"_a = nb::none(), "workers"_a = 1,
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
         "call returns.\n\n"
         "workers, keyword-only, is the number of threads the batch may use:\n"
         "1, the default, runs it in the calling thread; -1 uses every core\n"
         "available to the process (on Linux, its CPU affinity set). The\n"
         "results and their order do not depend on it. A batch too small to\n"
         "share uses fewer threads. Raises meshsearch.Error if workers is 0 or\n"
         "below -1, TypeError if it is not an integer.")

    .def("count_objects",
         [](const meshsearch::MeshGrid& self, const BatchArg& indices,
            const double rmax, const double rmin, const nb::object& out, const nb::object& workers) -> nb::object {
           const meshsearch::Workers threads = readWorkers(workers);
           const auto objects = readIndices(indices);
           const size_t queries = objects.size();
           OutArray target;
           std::int64_t* dest = nullptr;
           if (!out.is_none()) dest = outBuffer<std::int64_t>(target, out, {queries}, "int64");
           std::vector<std::uint64_t> found;
           {
             nb::gil_scoped_release released;
             runBatch(queries,
                      [&] { self.countObjects(objects, rmax, rmin, found, threads); },
                      [&] (const size_t i) { self.countObjects(objects[i], rmax, rmin); });
             if (dest != nullptr) copyCounts(found, dest);
           }
           if (dest != nullptr) return out;
           return nb::cast(int64Array(found));
         },
         "indices"_a, "rmax"_a, "rmin"_a = 0., nb::kw_only(), "out"_a = nb::none(), "workers"_a = 1,
         "For each object of an array of indices, the number of objects\n"
         "close_objects would return for it, the object itself excluded: an\n"
         "int64 array with one count per index, in order.\n\n"
         "indices is a 1-D array-like of integers, such as the uint32 arrays the\n"
         "queries return. An array of any integer dtype is read as it is,\n"
         "never cast, so values are never wrapped: a negative value or\n"
         "one that is not a valid index raises meshsearch.IndexError. A list or\n"
         "tuple of integers is converted first. Anything that is not integers\n"
         "raises TypeError; anything that is not one-dimensional raises\n"
         "meshsearch.Error.\n\n"
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
         "an index that is negative, out of range or removed, and returns nothing\n"
         "partial; meshsearch.Error on the radii, even for an empty\n"
         "array.\n\n"
         "The GIL is released while the queries run, so other Python threads\n"
         "may run, and may query the same grid; none may modify it until the\n"
         "call returns.\n\n"
         "workers, keyword-only, is the number of threads the batch may use:\n"
         "1, the default, runs it in the calling thread; -1 uses every core\n"
         "available to the process (on Linux, its CPU affinity set). The\n"
         "results and their order do not depend on it. A batch too small to\n"
         "share uses fewer threads. Raises meshsearch.Error if workers is 0 or\n"
         "below -1, TypeError if it is not an integer.")

    .def("get_cell", &meshsearch::MeshGrid::get_cell, "x"_a, "y"_a, "z"_a,
         "Linear index of the cell containing a point: (ix * ny + iy) * nz + iz,\n"
         "with (ix, iy, iz) from get_cell_coords and (nx, ny, nz) = n_cells.\n\n"
         "Raises meshsearch.Error if the point is outside the box.")

    .def("get_cell_coords",
         [](const meshsearch::MeshGrid& self, const double x, const double y, const double z) {
           return triple(self.get_cellCoords(x, y, z));
         },
         "x"_a, "y"_a, "z"_a,
         "Cell coordinates (ix, iy, iz) of the cell containing a point.\n\n"
         "Raises meshsearch.Error if the point is outside the box.")

    .def("get_objects_in_cell",
         [](const meshsearch::MeshGrid& self, const double x, const double y, const double z) {
           return indexArray(self.get_objectsInCell(x, y, z));
         },
         "x"_a, "y"_a, "z"_a,
         "Objects in the cell containing a point, as a uint32 array.\n\n"
         "Raises meshsearch.Error if the point is outside the box.")

    .def("remove_object",
         [](meshsearch::MeshGrid& self, const nb::object& index) { self.removeObject(scalarIndex(index)); },
         "index"_a,
         "Remove an object. Its index stays invalid for the life of the grid,\n"
         "and its memory is not returned. Must not run while another thread\n"
         "uses the grid.\n\n"
         "Raises meshsearch.IndexError if index is negative, out of range or\n"
         "already removed; TypeError if it is not an integer.")

    .def("add_object", &meshsearch::MeshGrid::addObject, "x"_a, "y"_a, "z"_a,
         "Add an object and return its index, one past the highest ever issued.\n"
         "The box is not refitted, and memory is not reclaimed: to add many\n"
         "objects, build a new grid. Must not run while another thread uses\n"
         "the grid.\n\n"
         "Raises meshsearch.Error if the point is outside the box, or if\n"
         "4294967295 indices have already been issued.");
}
