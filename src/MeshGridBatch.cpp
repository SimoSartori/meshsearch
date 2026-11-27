#include "meshsearch/MeshGrid.h"
#include "MeshGridWhole.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

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


struct meshsearch::MeshGrid::Grouped {

  struct Query {
    double x, y, z;
    size_t self;
    std::int64_t lo[3], hi[3];
  };

  static bool sort (const MeshGrid& g, const std::vector<unsigned int>& indices, const double Rmax, const double from,
                    std::vector<std::uint32_t>& order, std::vector<std::uint32_t>& bounds)
  {
    const size_t n = indices.size();
    const size_t nCells = g.m_cells.size();
    const size_t limit = std::numeric_limits<std::uint32_t>::max();
    if (indices.size() < cell_order_min || nCells == 0 || nCells >= limit || n >= limit || g.m_periodic_any
        || Rmax >= from) return false;
    for (const unsigned int index : indices) checkIndex(g.m_alive, index);
    std::vector<std::uint32_t> start(nCells+1, 0);
    std::vector<std::uint32_t> key(n);
    for (size_t i=0; i<n; ++i) {
      key[i] = static_cast<std::uint32_t>(g.objectCell(indices[i]));
      ++start[key[i]+1];
    }
    bounds.clear();
    bounds.push_back(0);
    for (size_t c=0; c<nCells; ++c) {
      if (start[c+1] != 0) bounds.push_back(start[c]+start[c+1]);
      start[c+1] += start[c];
    }
    order.resize(n);
    for (size_t i=0; i<n; ++i) order[start[key[i]]++] = static_cast<std::uint32_t>(i);
    return true;
  }

  template <class Hit>
  static void walk (const MeshGrid& g, const std::vector<unsigned int>& indices, const std::uint32_t* pos,
                    const size_t m, const double Rmax, const double Rmin, std::vector<Query>& queries, Hit&& hit)
  {
    const double Rmax_sq = Rmax*Rmax;
    const double Rmin_sq = Rmin*Rmin;
    const CellFaces faces = cellFaces(g.m_lims, g.m_nCells, g.m_cellsize);

    queries.resize(m);
    const size_t self0 = g.m_toInternal[indices[pos[0]]];
    const auto target = locate(g.m_lims, g.m_nCells, g.m_cellsize, g.m_X[self0], g.m_Y[self0], g.m_Z[self0]);
    StepWindow all = plainWindow(faces, target, g.m_nCells, g.m_X[self0], g.m_Y[self0], g.m_Z[self0]);
    all.update(Rmax_sq, true);
    for (size_t q=0; q<m; ++q) {
      Query& Q = queries[q];
      Q.self = g.m_toInternal[indices[pos[q]]];
      Q.x = g.m_X[Q.self];
      Q.y = g.m_Y[Q.self];
      Q.z = g.m_Z[Q.self];
      StepWindow w = plainWindow(faces, target, g.m_nCells, Q.x, Q.y, Q.z);
      w.update(Rmax_sq, true);
      for (int a=0; a<3; ++a) {
        Q.lo[a] = w.lo[a];
        Q.hi[a] = w.hi[a];
        all.lo[a] = std::min(all.lo[a], w.lo[a]);
        all.hi[a] = std::max(all.hi[a], w.hi[a]);
      }
    }

    const std::int64_t tx = target[0], ty = target[1], tz = target[2];
    const size_t ny = g.m_nCells[1], nz = g.m_nCells[2];
    const unsigned int nShells = static_cast<unsigned int>(g.m_mask->size());
    const unsigned int min_mask_idx = firstShell(faces, nShells, Rmin, Rmin_sq);

    for (unsigned int maskInd=min_mask_idx; maskInd<nShells; maskInd++) {
      if (shellBeyond(maskInd, g.m_cellsize, Rmax_sq)) break;
      ShellSegments<GridOffset> segments((*g.m_mask)[maskInd], (*g.m_runs)[maskInd], all, maskInd);
      while (segments.next()) {
        for (const GridOffset* entry=segments.begin; entry!=segments.end; ++entry) {
          if (!segments.holds(*entry)) continue;
          const std::int64_t currX = tx+entry->i;
          const std::int64_t currY = ty+entry->j;
          const std::int64_t currZ = tz+entry->k;
          const CellBlock& block = g.m_cells[(size_t(currX)*ny+size_t(currY))*nz+size_t(currZ)];
          if (block.count == 0) continue;
          for (size_t q=0; q<m; ++q) {
            const Query& Q = queries[q];
            if (entry->i < Q.lo[0] || entry->i > Q.hi[0] || entry->j < Q.lo[1] || entry->j > Q.hi[1]
                || entry->k < Q.lo[2] || entry->k > Q.hi[2]) continue;
            if (cellBeyond(faces, currX, currY, currZ, Q.x, Q.y, Q.z, Rmax_sq)) continue;
            hit(q, block.start, block.count, Rmax_sq, Rmin_sq);
          }
        }
      }
    }
  }

  static void count (const MeshGrid& g, const std::vector<unsigned int>& indices, const double Rmax, const double Rmin,
                     const std::vector<std::uint32_t>& order, const std::vector<std::uint32_t>& bounds,
                     std::vector<std::uint64_t>& out)
  {
    std::vector<Query> queries;
    std::vector<size_t> counts;
    for (size_t b=0; b+1<bounds.size(); ++b) {
      const std::uint32_t* pos = order.data()+bounds[b];
      const size_t m = bounds[b+1]-bounds[b];
      counts.assign(m, 0);
      walk(g, indices, pos, m, Rmax, Rmin, queries,
           [&] (const size_t q, const size_t first, const size_t n, const double Rmax_sq, const double Rmin_sq) {
             const Query& Q = queries[q];
             const size_t self = Q.self;
             size_t hits = 0;
             scanBlock(g.m_X.data()+first, g.m_Y.data()+first, g.m_Z.data()+first, n, Q.x, Q.y, Q.z,
                       [&] (const size_t k, const double dist_sq) {
                         if ((first+k != self) & (dist_sq <= Rmax_sq) & (dist_sq >= Rmin_sq)) ++hits;
                       });
             counts[q] += hits;
           });
      for (size_t q=0; q<m; ++q) out[pos[q]] = counts[q];
    }
  }

  static void close (const MeshGrid& g, const std::vector<unsigned int>& indices, const double Rmax, const double Rmin,
                     const std::vector<std::uint32_t>& order, const std::vector<std::uint32_t>& bounds, Batch& out)
  {
    const size_t n = indices.size();
    out.offsets.clear();
    out.offsets.resize(n+1, 0);
    out.indices.clear();
    std::vector<std::vector<unsigned int>> chunks(1);
    chunks.back().reserve(chunk_size);
    std::vector<std::uint32_t> chunk(n);
    std::vector<size_t> start(n);
    std::vector<Query> queries;
    std::vector<std::vector<unsigned int>> bufs;
    for (size_t b=0; b+1<bounds.size(); ++b) {
      const std::uint32_t* pos = order.data()+bounds[b];
      const size_t m = bounds[b+1]-bounds[b];
      if (bufs.size() < m) bufs.resize(m);
      for (size_t q=0; q<m; ++q) bufs[q].clear();
      walk(g, indices, pos, m, Rmax, Rmin, queries,
           [&] (const size_t q, const size_t first, const size_t count, const double Rmax_sq, const double Rmin_sq) {
             const Query& Q = queries[q];
             const size_t self = Q.self;
             std::vector<unsigned int>& found = bufs[q];
             const unsigned int* pub = g.m_toPublic.data()+first;
             scanBlock(g.m_X.data()+first, g.m_Y.data()+first, g.m_Z.data()+first, count, Q.x, Q.y, Q.z,
                       [&] (const size_t k, const double dist_sq) {
                         if (first+k != self && dist_sq <= Rmax_sq && dist_sq >= Rmin_sq) found.push_back(pub[k]);
                       });
           });
      for (size_t q=0; q<m; ++q) {
        const size_t i = pos[q];
        if (chunks.back().capacity()-chunks.back().size() < chunk_margin) {
          chunks.emplace_back();
          chunks.back().reserve(chunk_size);
        }
        std::vector<unsigned int>& buffer = chunks.back();
        chunk[i] = static_cast<std::uint32_t>(chunks.size()-1);
        start[i] = buffer.size();
        buffer.insert(buffer.end(), bufs[q].begin(), bufs[q].end());
        out.offsets[i+1] = bufs[q].size();
      }
    }
    for (size_t i=0; i<n; ++i) out.offsets[i+1] += out.offsets[i];
    out.indices.reserve(out.offsets[n]);
    for (size_t i=0; i<n; ++i) {
      const unsigned int* first = chunks[chunk[i]].data()+start[i];
      out.indices.insert(out.indices.end(), first, first+(out.offsets[i+1]-out.offsets[i]));
    }
  }

};


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
  if (&out.indices == &indices) {
    const std::vector<unsigned int> copy(indices);
    closeObjects(copy, Rmax, Rmin, out);
    return;
  }

  checkRadii(Rmax, Rmin);

  std::vector<std::uint32_t> order, bounds;
  if (Grouped::sort(*this, indices, Rmax, m_closeFrom, order, bounds))
    return Grouped::close(*this, indices, Rmax, Rmin, order, bounds, out);

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
  std::vector<std::uint32_t> order, bounds;
  if (Grouped::sort(*this, indices, Rmax, m_countFrom, order, bounds))
    return Grouped::count(*this, indices, Rmax, Rmin, order, bounds, out);

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
