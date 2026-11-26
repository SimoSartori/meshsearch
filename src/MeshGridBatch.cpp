#include "meshsearch/MeshGrid.h"
#include "MeshGridWhole.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace {

  const size_t cell_order_min = 100000;

  const size_t chunk_size = size_t(1) << 20;

  const size_t chunk_margin = size_t(1) << 15;

  template <class Key, class Run>
  void runQueries (const size_t n, const size_t nCells, Key&& key, Run&& run)
  {
    if (n < cell_order_min || nCells == 0) {
      for (size_t i=0; i<n; ++i) run(i);
      return;
    }
    const size_t width = nCells/n+1;
    const size_t nBuckets = nCells/width+1;
    std::vector<size_t> bucket(n);
    std::vector<size_t> start(nBuckets+1, 0);
    for (size_t i=0; i<n; ++i) {
      bucket[i] = key(i)/width;
      ++start[bucket[i]+1];
    }
    for (size_t b=0; b<nBuckets; ++b) start[b+1] += start[b];
    std::vector<size_t> order(n);
    for (size_t i=0; i<n; ++i) order[start[bucket[i]]++] = i;
    for (const size_t i : order) run(i);
  }

  template <class Key, class Scan, class Append>
  void closeBatch (meshsearch::MeshGrid::Batch& out, const size_t n, const size_t nCells, Key&& key, Scan&& scan,
                   Append&& append)
  {
    out.offsets.clear();
    out.offsets.reserve(n+1);
    out.offsets.push_back(0);

    if (n < cell_order_min || nCells == 0) {
      std::vector<unsigned int> found;
      found.swap(out.indices);
      found.clear();
      std::vector<unsigned int> one;
      for (size_t i=0; i<n; ++i) {
        scan(one, i);
        found.insert(found.end(), one.begin(), one.end());
        out.offsets.push_back(found.size());
      }
      out.indices.swap(found);
      return;
    }

    out.indices.clear();
    out.offsets.resize(n+1);
    std::vector<std::vector<unsigned int>> chunks(1);
    chunks.back().reserve(chunk_size);
    std::vector<std::uint32_t> chunk(n);
    std::vector<size_t> start(n);
    runQueries(n, nCells, key,
               [&] (const size_t i) {
                 if (chunks.back().capacity()-chunks.back().size() < chunk_margin) {
                   chunks.emplace_back();
                   chunks.back().reserve(chunk_size);
                 }
                 std::vector<unsigned int>& buffer = chunks.back();
                 chunk[i] = static_cast<std::uint32_t>(chunks.size()-1);
                 start[i] = buffer.size();
                 append(buffer, i);
                 out.offsets[i+1] = buffer.size()-start[i];
               });
    for (size_t i=0; i<n; ++i) out.offsets[i+1] += out.offsets[i];
    out.indices.reserve(out.offsets[n]);
    for (size_t i=0; i<n; ++i) {
      const unsigned int* first = chunks[chunk[i]].data()+start[i];
      out.indices.insert(out.indices.end(), first, first+(out.offsets[i+1]-out.offsets[i]));
    }
  }

  void checkLengths (const std::vector<double>& X, const std::vector<double>& Y, const std::vector<double>& Z)
  {
    if (X.size() != Y.size() || X.size() != Z.size()) throw meshsearch::Error("the coordinate vectors differ in length");
  }

  size_t batchSize (const size_t n, const unsigned int N, const size_t max_size)
  {
    if (N != 0 && n > max_size/N) throw meshsearch::Error("the batch result is too large");
    return n*N;
  }

}


// ===========================================================================


std::vector<unsigned int> meshsearch::MeshGrid::nearestObject (const std::vector<double>& X,
                                                               const std::vector<double>& Y,
                                                               const std::vector<double>& Z) const
{
  std::vector<unsigned int> out;
  nearestObject(X, Y, Z, out);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::nearestObject (const std::vector<double>& X,
                                          const std::vector<double>& Y,
                                          const std::vector<double>& Z,
                                          std::vector<unsigned int>& out) const
{
  checkLengths(X, Y, Z);
  if (m_nObjects == 0) throw Error("the grid holds no objects");

  out.clear();
  out.resize(X.size());
  runQueries(X.size(), m_cells.size(),
             [&] (const size_t i) { return get_cell(X[i], Y[i], Z[i]); },
             [&] (const size_t i) { out[i] = nearestObject(X[i], Y[i], Z[i]); });
}


// ===========================================================================


std::vector<unsigned int> meshsearch::MeshGrid::nearestObject (const std::vector<unsigned int>& indices) const
{
  std::vector<unsigned int> out;
  nearestObject(indices, out);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::nearestObject (const std::vector<unsigned int>& indices,
                                          std::vector<unsigned int>& out) const
{
  if (&out == &indices) {
    std::vector<unsigned int> result;
    nearestObject(indices, result);
    out.swap(result);
    return;
  }

  if (m_nObjects < 2) throw Error("no other object remains in the grid");

  out.clear();
  out.resize(indices.size());
  runQueries(indices.size(), m_cells.size(),
             [&] (const size_t i) { return objectCell(indices[i]); },
             [&] (const size_t i) { out[i] = nearestObject(indices[i]); });
}


// ===========================================================================


std::vector<unsigned int> meshsearch::MeshGrid::nearestObjects (const unsigned int N,
                                                                const std::vector<double>& X,
                                                                const std::vector<double>& Y,
                                                                const std::vector<double>& Z) const
{
  std::vector<unsigned int> out;
  nearestObjects(N, X, Y, Z, out);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::nearestObjects (const unsigned int N,
                                           const std::vector<double>& X,
                                           const std::vector<double>& Y,
                                           const std::vector<double>& Z,
                                           std::vector<unsigned int>& out) const
{
  checkLengths(X, Y, Z);
  if (N > m_nObjects) throw Error("N is larger than the number of objects");

  out.clear();
  out.resize(batchSize(X.size(), N, out.max_size()));
  std::vector<unsigned int> one;
  runQueries(X.size(), m_cells.size(),
             [&] (const size_t i) { return get_cell(X[i], Y[i], Z[i]); },
             [&] (const size_t i) {
               nearestObjects(N, X[i], Y[i], Z[i], one);
               std::copy(one.begin(), one.end(), out.begin()+static_cast<std::ptrdiff_t>(i*N));
             });
}


// ===========================================================================


std::vector<unsigned int> meshsearch::MeshGrid::nearestObjects (const unsigned int N,
                                                                const std::vector<unsigned int>& indices) const
{
  std::vector<unsigned int> out;
  nearestObjects(N, indices, out);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::nearestObjects (const unsigned int N, const std::vector<unsigned int>& indices,
                                           std::vector<unsigned int>& out) const
{
  if (&out == &indices) {
    std::vector<unsigned int> result;
    nearestObjects(N, indices, result);
    out.swap(result);
    return;
  }

  if (N >= m_nObjects) throw Error("N is larger than the number of available neighbours");

  out.clear();
  out.resize(batchSize(indices.size(), N, out.max_size()));
  std::vector<unsigned int> one;
  runQueries(indices.size(), m_cells.size(),
             [&] (const size_t i) { return objectCell(indices[i]); },
             [&] (const size_t i) {
               nearestObjects(N, indices[i], one);
               std::copy(one.begin(), one.end(), out.begin()+static_cast<std::ptrdiff_t>(i*N));
             });
}


// ===========================================================================


meshsearch::MeshGrid::Batch meshsearch::MeshGrid::closeObjects (const std::vector<double>& X,
                                                                const std::vector<double>& Y,
                                                                const std::vector<double>& Z,
                                                                const double Rmax, const double Rmin) const
{
  Batch out;
  closeObjects(X, Y, Z, Rmax, Rmin, out);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::closeObjects (const std::vector<double>& X,
                                         const std::vector<double>& Y,
                                         const std::vector<double>& Z,
                                         const double Rmax, const double Rmin, Batch& out) const
{
  checkLengths(X, Y, Z);
  checkRadii(Rmax, Rmin);

  closeBatch(out, X.size(), m_cells.size(),
             [&] (const size_t i) { return get_cell(X[i], Y[i], Z[i]); },
             [&] (std::vector<unsigned int>& one, const size_t i) { closeObjects(X[i], Y[i], Z[i], Rmax, Rmin, one); },
             [&] (std::vector<unsigned int>& buffer, const size_t i) { closeAppend(buffer, X[i], Y[i], Z[i], Rmax, Rmin, nullptr); });
}


// ===========================================================================


void meshsearch::MeshGrid::closeObjects (const std::vector<double>& X,
                                         const std::vector<double>& Y,
                                         const std::vector<double>& Z,
                                         const double Rmax, Batch& out) const
{
  closeObjects(X, Y, Z, Rmax, 0., out);
}


// ===========================================================================


meshsearch::MeshGrid::Batch meshsearch::MeshGrid::closeObjects (const std::vector<unsigned int>& indices,
                                                                const double Rmax, const double Rmin) const
{
  Batch out;
  closeObjects(indices, Rmax, Rmin, out);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::closeObjects (const std::vector<unsigned int>& indices,
                                         const double Rmax, const double Rmin, Batch& out) const
{
  checkRadii(Rmax, Rmin);

  closeBatch(out, indices.size(), m_cells.size(),
             [&] (const size_t i) { return objectCell(indices[i]); },
             [&] (std::vector<unsigned int>& one, const size_t i) { closeObjects(indices[i], Rmax, Rmin, one); },
             [&] (std::vector<unsigned int>& buffer, const size_t i) { closeAppend(buffer, 0., 0., 0., Rmax, Rmin, &indices[i]); });
}


// ===========================================================================


void meshsearch::MeshGrid::closeObjects (const std::vector<unsigned int>& indices,
                                         const double Rmax, Batch& out) const
{
  closeObjects(indices, Rmax, 0., out);
}


// ===========================================================================


std::vector<std::uint64_t> meshsearch::MeshGrid::countObjects (const std::vector<double>& X,
                                                               const std::vector<double>& Y,
                                                               const std::vector<double>& Z,
                                                               const double Rmax, const double Rmin) const
{
  std::vector<std::uint64_t> out;
  countObjects(X, Y, Z, Rmax, Rmin, out);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::countObjects (const std::vector<double>& X,
                                         const std::vector<double>& Y,
                                         const std::vector<double>& Z,
                                         const double Rmax, const double Rmin,
                                         std::vector<std::uint64_t>& out) const
{
  checkLengths(X, Y, Z);
  checkRadii(Rmax, Rmin);

  out.clear();
  out.resize(X.size());
  runQueries(X.size(), m_cells.size(),
             [&] (const size_t i) { return get_cell(X[i], Y[i], Z[i]); },
             [&] (const size_t i) { out[i] = countObjects(X[i], Y[i], Z[i], Rmax, Rmin); });
}


// ===========================================================================


void meshsearch::MeshGrid::countObjects (const std::vector<double>& X,
                                         const std::vector<double>& Y,
                                         const std::vector<double>& Z,
                                         const double Rmax, std::vector<std::uint64_t>& out) const
{
  countObjects(X, Y, Z, Rmax, 0., out);
}


// ===========================================================================


std::vector<std::uint64_t> meshsearch::MeshGrid::countObjects (const std::vector<unsigned int>& indices,
                                                               const double Rmax, const double Rmin) const
{
  std::vector<std::uint64_t> out;
  countObjects(indices, Rmax, Rmin, out);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::countObjects (const std::vector<unsigned int>& indices,
                                         const double Rmax, const double Rmin,
                                         std::vector<std::uint64_t>& out) const
{
  checkRadii(Rmax, Rmin);

  out.clear();
  out.resize(indices.size());
  runQueries(indices.size(), m_cells.size(),
             [&] (const size_t i) { return objectCell(indices[i]); },
             [&] (const size_t i) { out[i] = countObjects(indices[i], Rmax, Rmin); });
}


// ===========================================================================


void meshsearch::MeshGrid::countObjects (const std::vector<unsigned int>& indices,
                                         const double Rmax, std::vector<std::uint64_t>& out) const
{
  countObjects(indices, Rmax, 0., out);
}


// ===========================================================================


size_t meshsearch::MeshGrid::objectCell (const unsigned int index) const
{
  if (!isAlive(index)) return 0;
  const size_t slot = m_toInternal[index];
  return get_cell(m_X[slot], m_Y[slot], m_Z[slot]);
}


// ===========================================================================


std::vector<std::uint8_t> meshsearch::MeshGrid::isAlive (const std::vector<unsigned int>& indices) const
{
  std::vector<std::uint8_t> out;
  isAlive(indices, out);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::isAlive (const std::vector<unsigned int>& indices, std::vector<std::uint8_t>& out) const
{
  out.clear();
  out.reserve(indices.size());
  for (const unsigned int index : indices) out.push_back(isAlive(index) ? 1 : 0);
}


// ===========================================================================


void meshsearch::MeshGrid::closeAppend (std::vector<unsigned int>& out, double X, double Y, double Z,
                                        const double Rmax, const double Rmin, const unsigned int* index) const
{
  size_t self = no_slot;
  if (index) {
    checkIndex(m_alive, *index);
    self = m_toInternal[*index];
    X = m_X[self];
    Y = m_Y[self];
    Z = m_Z[self];
  }
  const auto target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);

  const double Rmax_sq = Rmax*Rmax;
  const double Rmin_sq = Rmin*Rmin;

  std::vector<unsigned int> found;
  found.swap(out);

  auto run = [&] (auto exclude) {
    auto each = [&] (const size_t first, const size_t k, const double dist_sq) {
      if ((!decltype(exclude)::value || first+k != self) && dist_sq <= Rmax_sq && dist_sq >= Rmin_sq)
        found.push_back(m_toPublic[first+k]);
    };
    if (Rmax >= m_closeFrom)
      walk(target, X, Y, Z, Rmin, [&] { return Rmax_sq; },
           wholeVisit(wholeTest(Rmax, Rmin, m_cellsize), each,
                      [&] (const size_t first, const size_t count) {
                        emitBlock(m_toPublic.data()+first, first, count, self,
                                  [&] (const unsigned int* begin, const unsigned int* end) {
                                    found.insert(found.end(), begin, end);
                                  });
                      }));
    else walk(target, X, Y, Z, Rmin, [&] { return Rmax_sq; }, each);
  };
  if (index) run(std::true_type());
  else run(std::false_type());

  out.swap(found);
}
