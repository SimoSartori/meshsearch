#include "meshsearch/MeshGrid.h"
#include "MeshGridWhole.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

namespace {

  const size_t cell_order_min = 100000;

  const size_t chunk_size = size_t(1) << 20;

  const size_t chunk_margin = size_t(1) << 15;

  const size_t thread_min = 256;

  const size_t thread_block = 64;

  template <class Key>
  std::vector<size_t> cellOrder (const size_t n, const size_t nCells, Key&& key)
  {
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
    return order;
  }

  template <class Key, class Run>
  void runQueries (const size_t n, const size_t nCells, Key&& key, Run&& run)
  {
    if (n < cell_order_min || nCells == 0) {
      for (size_t i=0; i<n; ++i) run(i);
      return;
    }
    for (const size_t i : cellOrder(n, nCells, key)) run(i);
  }

  size_t available ()
  {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0 && CPU_COUNT(&set) > 0) return size_t(CPU_COUNT(&set));
#endif
    const unsigned int cores = std::thread::hardware_concurrency();
    return cores > 0 ? cores : 1;
  }

  [[gnu::noinline]] size_t threadsFor (const meshsearch::Workers workers, const size_t n)
  {
    if (workers.get_count() == 1 || n < 2*thread_min) return 1;
    const size_t asked = workers.get_count() == -1 ? available() : size_t(workers.get_count());
    return std::max<size_t>(1, std::min(asked, n/thread_min));
  }

  template <class Body>
  void together (const size_t threads, Body&& body)
  {
    std::atomic<bool> failed(false);
    std::exception_ptr error;
    std::mutex lock;
    auto guarded = [&] () {
      try {
        body(failed);
      }
      catch (...) {
        const std::lock_guard<std::mutex> hold(lock);
        if (!error) error = std::current_exception();
        failed.store(true, std::memory_order_relaxed);
      }
    };
    std::vector<std::thread> pool;
    try {
      pool.reserve(threads-1);
      for (size_t t=1; t<threads; ++t) pool.emplace_back(guarded);
    }
    catch (...) {}
    guarded();
    for (std::thread& thread : pool) thread.join();
    if (error) std::rethrow_exception(error);
  }

  template <class Make>
  void spread (const size_t n, const size_t block, const size_t threads, Make&& make)
  {
    std::atomic<size_t> next(0);
    together(threads, [&] (const std::atomic<bool>& failed) {
      auto run = make();
      while (!failed.load(std::memory_order_relaxed)) {
        const size_t first = next.fetch_add(block, std::memory_order_relaxed);
        if (first >= n) return;
        const size_t last = std::min(n, first+block);
        for (size_t p=first; p<last; ++p) run(p);
      }
    });
  }

  template <class F>
  [[gnu::noinline]] void offline (F&& f)
  {
    f();
  }

  template <class Key, class Make>
  void runParallel (const size_t n, const size_t nCells, const size_t threads, Key&& key, Make&& make)
  {
    if (n < cell_order_min || nCells == 0) return spread(n, thread_block, threads, make);
    const std::vector<size_t> order = cellOrder(n, nCells, key);
    spread(n, thread_block, threads, [&] {
      return [&order, run = make()] (const size_t p) mutable { run(order[p]); };
    });
  }

  class Chunks {
  public:
    std::vector<unsigned int>& room (std::uint32_t& id, std::vector<unsigned int>*& current)
    {
      if (current == nullptr || current->capacity()-current->size() < chunk_margin) {
        {
          const std::lock_guard<std::mutex> hold(m_lock);
          m_list.emplace_back();
          current = &m_list.back();
          id = static_cast<std::uint32_t>(m_list.size()-1);
        }
        current->reserve(chunk_size);
      }
      return *current;
    }
    const std::vector<unsigned int>& operator[] (const std::uint32_t id) const { return m_list[id]; }
  private:
    std::deque<std::vector<unsigned int>> m_list;
    std::mutex m_lock;
  };

  void gather (meshsearch::MeshGrid::Batch& out, const size_t n, const Chunks& chunks,
               const std::vector<std::uint32_t>& chunk, const std::vector<size_t>& start)
  {
    for (size_t i=0; i<n; ++i) out.offsets[i+1] += out.offsets[i];
    out.indices.reserve(out.offsets[n]);
    for (size_t i=0; i<n; ++i) {
      const unsigned int* first = chunks[chunk[i]].data()+start[i];
      out.indices.insert(out.indices.end(), first, first+(out.offsets[i+1]-out.offsets[i]));
    }
  }

  template <class Key, class Append>
  void closeParallel (meshsearch::MeshGrid::Batch& out, const size_t n, const size_t nCells, const size_t threads,
                      Key&& key, Append&& append)
  {
    out.offsets.clear();
    out.offsets.resize(n+1, 0);
    out.indices.clear();
    Chunks chunks;
    std::vector<std::uint32_t> chunk(n);
    std::vector<size_t> start(n);
    runParallel(n, nCells, threads, key, [&] {
      return [&, id = std::uint32_t(0), current = static_cast<std::vector<unsigned int>*>(nullptr)] (const size_t i) mutable {
        std::vector<unsigned int>& buffer = chunks.room(id, current);
        chunk[i] = id;
        start[i] = buffer.size();
        append(buffer, i);
        out.offsets[i+1] = buffer.size()-start[i];
      };
    });
    gather(out, n, chunks, chunk, start);
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

  void checkPoints (const Lims& lims, const std::array<unsigned int, 3>& nCells, const double cellsize,
                    const std::vector<double>& X, const std::vector<double>& Y, const std::vector<double>& Z)
  {
    for (size_t i=0; i<X.size(); ++i) locate(lims, nCells, cellsize, X[i], Y[i], Z[i]);
  }

  void checkIndices (const std::vector<char>& alive, const std::vector<unsigned int>& indices)
  {
    for (const unsigned int index : indices) checkIndex(alive, index);
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

  static void countGroup (const MeshGrid& g, const std::vector<unsigned int>& indices, const std::uint32_t* pos,
                          const size_t m, const double Rmax, const double Rmin, std::vector<Query>& queries,
                          std::vector<size_t>& counts, std::vector<std::uint64_t>& out)
  {
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

  static void count (const MeshGrid& g, const std::vector<unsigned int>& indices, const double Rmax, const double Rmin,
                     const std::vector<std::uint32_t>& order, const std::vector<std::uint32_t>& bounds,
                     const size_t threads, std::vector<std::uint64_t>& out)
  {
    if (threads > 1)
      return spread(bounds.size()-1, 1, threads, [&] {
        return [&, queries = std::vector<Query>(), counts = std::vector<size_t>()] (const size_t b) mutable {
          countGroup(g, indices, order.data()+bounds[b], bounds[b+1]-bounds[b], Rmax, Rmin, queries, counts, out);
        };
      });
    std::vector<Query> queries;
    std::vector<size_t> counts;
    for (size_t b=0; b+1<bounds.size(); ++b)
      countGroup(g, indices, order.data()+bounds[b], bounds[b+1]-bounds[b], Rmax, Rmin, queries, counts, out);
  }

  static void closeGroup (const MeshGrid& g, const std::vector<unsigned int>& indices, const std::uint32_t* pos,
                          const size_t m, const double Rmax, const double Rmin, std::vector<Query>& queries,
                          std::vector<std::vector<unsigned int>>& bufs)
  {
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
  }

  static void close (const MeshGrid& g, const std::vector<unsigned int>& indices, const double Rmax, const double Rmin,
                     const std::vector<std::uint32_t>& order, const std::vector<std::uint32_t>& bounds,
                     const size_t threads, Batch& out)
  {
    const size_t n = indices.size();
    if (threads > 1) {
      out.offsets.clear();
      out.offsets.resize(n+1, 0);
      out.indices.clear();
      Chunks chunks;
      std::vector<std::uint32_t> chunk(n);
      std::vector<size_t> start(n);
      spread(bounds.size()-1, 1, threads, [&] {
        return [&, queries = std::vector<Query>(), bufs = std::vector<std::vector<unsigned int>>(),
                id = std::uint32_t(0), current = static_cast<std::vector<unsigned int>*>(nullptr)] (const size_t b) mutable {
          const std::uint32_t* pos = order.data()+bounds[b];
          const size_t m = bounds[b+1]-bounds[b];
          closeGroup(g, indices, pos, m, Rmax, Rmin, queries, bufs);
          for (size_t q=0; q<m; ++q) {
            const size_t i = pos[q];
            std::vector<unsigned int>& buffer = chunks.room(id, current);
            chunk[i] = id;
            start[i] = buffer.size();
            buffer.insert(buffer.end(), bufs[q].begin(), bufs[q].end());
            out.offsets[i+1] = bufs[q].size();
          }
        };
      });
      gather(out, n, chunks, chunk, start);
      return;
    }
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
      closeGroup(g, indices, pos, m, Rmax, Rmin, queries, bufs);
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


meshsearch::Workers::Workers (const int count)
  : m_count(count)
{
  if (count < 1 && count != -1) throw Error("workers must be at least 1, or -1 for every available core");
}


// ===========================================================================


std::vector<unsigned int> meshsearch::MeshGrid::nearestObject (const std::vector<double>& X,
                                                               const std::vector<double>& Y,
                                                               const std::vector<double>& Z,
                                                               const Workers workers) const
{
  std::vector<unsigned int> out;
  nearestObject(X, Y, Z, out, workers);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::nearestObject (const std::vector<double>& X,
                                          const std::vector<double>& Y,
                                          const std::vector<double>& Z,
                                          std::vector<unsigned int>& out,
                                          const Workers workers) const
{
  checkLengths(X, Y, Z);
  if (m_nObjects == 0) throw Error("the grid holds no objects");

  out.clear();
  out.resize(X.size());
  const size_t threads = threadsFor(workers, X.size());
  if (threads > 1)
    return offline([&] {
      checkPoints(m_lims, m_nCells, m_cellsize, X, Y, Z);
      runParallel(X.size(), m_cells.size(), threads,
                  [&] (const size_t i) { return get_cell(X[i], Y[i], Z[i]); },
                  [&] { return [&] (const size_t i) { out[i] = nearestObject(X[i], Y[i], Z[i]); }; });
    });
  runQueries(X.size(), m_cells.size(),
             [&] (const size_t i) { return get_cell(X[i], Y[i], Z[i]); },
             [&] (const size_t i) { out[i] = nearestObject(X[i], Y[i], Z[i]); });
}


// ===========================================================================


std::vector<unsigned int> meshsearch::MeshGrid::nearestObject (const std::vector<unsigned int>& indices,
                                                               const Workers workers) const
{
  std::vector<unsigned int> out;
  nearestObject(indices, out, workers);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::nearestObject (const std::vector<unsigned int>& indices,
                                          std::vector<unsigned int>& out,
                                          const Workers workers) const
{
  if (&out == &indices) {
    std::vector<unsigned int> result;
    nearestObject(indices, result, workers);
    out.swap(result);
    return;
  }

  if (m_nObjects < 2) throw Error("no other object remains in the grid");

  out.clear();
  out.resize(indices.size());
  const size_t threads = threadsFor(workers, indices.size());
  if (threads > 1)
    return offline([&] {
      checkIndices(m_alive, indices);
      runParallel(indices.size(), m_cells.size(), threads,
                  [&] (const size_t i) { return objectCell(indices[i]); },
                  [&] { return [&] (const size_t i) { out[i] = nearestObject(indices[i]); }; });
    });
  runQueries(indices.size(), m_cells.size(),
             [&] (const size_t i) { return objectCell(indices[i]); },
             [&] (const size_t i) { out[i] = nearestObject(indices[i]); });
}


// ===========================================================================


std::vector<unsigned int> meshsearch::MeshGrid::nearestObjects (const unsigned int N,
                                                                const std::vector<double>& X,
                                                                const std::vector<double>& Y,
                                                                const std::vector<double>& Z,
                                                                const Workers workers) const
{
  std::vector<unsigned int> out;
  nearestObjects(N, X, Y, Z, out, workers);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::nearestObjects (const unsigned int N,
                                           const std::vector<double>& X,
                                           const std::vector<double>& Y,
                                           const std::vector<double>& Z,
                                           std::vector<unsigned int>& out,
                                           const Workers workers) const
{
  checkLengths(X, Y, Z);
  if (N > m_nObjects) throw Error("N is larger than the number of objects");

  out.clear();
  out.resize(batchSize(X.size(), N, out.max_size()));
  const size_t threads = threadsFor(workers, X.size());
  if (threads > 1)
    return offline([&] {
      checkPoints(m_lims, m_nCells, m_cellsize, X, Y, Z);
      runParallel(X.size(), m_cells.size(), threads,
                  [&] (const size_t i) { return get_cell(X[i], Y[i], Z[i]); },
                  [&] {
                    return [&, one = std::vector<unsigned int>()] (const size_t i) mutable {
                      nearestObjects(N, X[i], Y[i], Z[i], one);
                      std::copy(one.begin(), one.end(), out.begin()+static_cast<std::ptrdiff_t>(i*N));
                    };
                  });
    });
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
                                                                const std::vector<unsigned int>& indices,
                                                                const Workers workers) const
{
  std::vector<unsigned int> out;
  nearestObjects(N, indices, out, workers);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::nearestObjects (const unsigned int N, const std::vector<unsigned int>& indices,
                                           std::vector<unsigned int>& out, const Workers workers) const
{
  if (&out == &indices) {
    std::vector<unsigned int> result;
    nearestObjects(N, indices, result, workers);
    out.swap(result);
    return;
  }

  if (N >= m_nObjects) throw Error("N is larger than the number of available neighbours");

  out.clear();
  out.resize(batchSize(indices.size(), N, out.max_size()));
  const size_t threads = threadsFor(workers, indices.size());
  if (threads > 1)
    return offline([&] {
      checkIndices(m_alive, indices);
      runParallel(indices.size(), m_cells.size(), threads,
                  [&] (const size_t i) { return objectCell(indices[i]); },
                  [&] {
                    return [&, one = std::vector<unsigned int>()] (const size_t i) mutable {
                      nearestObjects(N, indices[i], one);
                      std::copy(one.begin(), one.end(), out.begin()+static_cast<std::ptrdiff_t>(i*N));
                    };
                  });
    });
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
                                                                const double Rmax, const double Rmin,
                                                                const Workers workers) const
{
  Batch out;
  closeObjects(X, Y, Z, Rmax, Rmin, out, workers);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::closeObjects (const std::vector<double>& X,
                                         const std::vector<double>& Y,
                                         const std::vector<double>& Z,
                                         const double Rmax, const double Rmin, Batch& out,
                                         const Workers workers) const
{
  checkLengths(X, Y, Z);
  checkRadii(Rmax, Rmin);

  const size_t threads = threadsFor(workers, X.size());
  if (threads > 1)
    return offline([&] {
      checkPoints(m_lims, m_nCells, m_cellsize, X, Y, Z);
      closeParallel(out, X.size(), m_cells.size(), threads,
                    [&] (const size_t i) { return get_cell(X[i], Y[i], Z[i]); },
                    [&] (std::vector<unsigned int>& buffer, const size_t i) { closeAppend(buffer, X[i], Y[i], Z[i], Rmax, Rmin, nullptr); });
    });

  closeBatch(out, X.size(), m_cells.size(),
             [&] (const size_t i) { return get_cell(X[i], Y[i], Z[i]); },
             [&] (std::vector<unsigned int>& one, const size_t i) { closeObjects(X[i], Y[i], Z[i], Rmax, Rmin, one); },
             [&] (std::vector<unsigned int>& buffer, const size_t i) { closeAppend(buffer, X[i], Y[i], Z[i], Rmax, Rmin, nullptr); });
}


// ===========================================================================


void meshsearch::MeshGrid::closeObjects (const std::vector<double>& X,
                                         const std::vector<double>& Y,
                                         const std::vector<double>& Z,
                                         const double Rmax, Batch& out, const Workers workers) const
{
  closeObjects(X, Y, Z, Rmax, 0., out, workers);
}


// ===========================================================================


meshsearch::MeshGrid::Batch meshsearch::MeshGrid::closeObjects (const std::vector<unsigned int>& indices,
                                                                const double Rmax, const double Rmin,
                                                                const Workers workers) const
{
  Batch out;
  closeObjects(indices, Rmax, Rmin, out, workers);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::closeObjects (const std::vector<unsigned int>& indices,
                                         const double Rmax, const double Rmin, Batch& out,
                                         const Workers workers) const
{
  if (&out.indices == &indices) {
    const std::vector<unsigned int> copy(indices);
    closeObjects(copy, Rmax, Rmin, out, workers);
    return;
  }

  checkRadii(Rmax, Rmin);

  const size_t threads = threadsFor(workers, indices.size());
  std::vector<std::uint32_t> order, bounds;
  if (Grouped::sort(*this, indices, Rmax, m_closeFrom, order, bounds))
    return Grouped::close(*this, indices, Rmax, Rmin, order, bounds, threads, out);

  if (threads > 1)
    return offline([&] {
      checkIndices(m_alive, indices);
      closeParallel(out, indices.size(), m_cells.size(), threads,
                    [&] (const size_t i) { return objectCell(indices[i]); },
                    [&] (std::vector<unsigned int>& buffer, const size_t i) { closeAppend(buffer, 0., 0., 0., Rmax, Rmin, &indices[i]); });
    });

  closeBatch(out, indices.size(), m_cells.size(),
             [&] (const size_t i) { return objectCell(indices[i]); },
             [&] (std::vector<unsigned int>& one, const size_t i) { closeObjects(indices[i], Rmax, Rmin, one); },
             [&] (std::vector<unsigned int>& buffer, const size_t i) { closeAppend(buffer, 0., 0., 0., Rmax, Rmin, &indices[i]); });
}


// ===========================================================================


void meshsearch::MeshGrid::closeObjects (const std::vector<unsigned int>& indices,
                                         const double Rmax, Batch& out, const Workers workers) const
{
  closeObjects(indices, Rmax, 0., out, workers);
}


// ===========================================================================


std::vector<std::uint64_t> meshsearch::MeshGrid::countObjects (const std::vector<double>& X,
                                                               const std::vector<double>& Y,
                                                               const std::vector<double>& Z,
                                                               const double Rmax, const double Rmin,
                                                               const Workers workers) const
{
  std::vector<std::uint64_t> out;
  countObjects(X, Y, Z, Rmax, Rmin, out, workers);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::countObjects (const std::vector<double>& X,
                                         const std::vector<double>& Y,
                                         const std::vector<double>& Z,
                                         const double Rmax, const double Rmin,
                                         std::vector<std::uint64_t>& out, const Workers workers) const
{
  checkLengths(X, Y, Z);
  checkRadii(Rmax, Rmin);

  out.clear();
  out.resize(X.size());
  const size_t threads = threadsFor(workers, X.size());
  if (threads > 1)
    return offline([&] {
      checkPoints(m_lims, m_nCells, m_cellsize, X, Y, Z);
      runParallel(X.size(), m_cells.size(), threads,
                  [&] (const size_t i) { return get_cell(X[i], Y[i], Z[i]); },
                  [&] { return [&] (const size_t i) { out[i] = countObjects(X[i], Y[i], Z[i], Rmax, Rmin); }; });
    });
  runQueries(X.size(), m_cells.size(),
             [&] (const size_t i) { return get_cell(X[i], Y[i], Z[i]); },
             [&] (const size_t i) { out[i] = countObjects(X[i], Y[i], Z[i], Rmax, Rmin); });
}


// ===========================================================================


void meshsearch::MeshGrid::countObjects (const std::vector<double>& X,
                                         const std::vector<double>& Y,
                                         const std::vector<double>& Z,
                                         const double Rmax, std::vector<std::uint64_t>& out,
                                         const Workers workers) const
{
  countObjects(X, Y, Z, Rmax, 0., out, workers);
}


// ===========================================================================


std::vector<std::uint64_t> meshsearch::MeshGrid::countObjects (const std::vector<unsigned int>& indices,
                                                               const double Rmax, const double Rmin,
                                                               const Workers workers) const
{
  std::vector<std::uint64_t> out;
  countObjects(indices, Rmax, Rmin, out, workers);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::countObjects (const std::vector<unsigned int>& indices,
                                         const double Rmax, const double Rmin,
                                         std::vector<std::uint64_t>& out, const Workers workers) const
{
  checkRadii(Rmax, Rmin);

  out.clear();
  out.resize(indices.size());
  const size_t threads = threadsFor(workers, indices.size());
  std::vector<std::uint32_t> order, bounds;
  if (Grouped::sort(*this, indices, Rmax, m_countFrom, order, bounds))
    return Grouped::count(*this, indices, Rmax, Rmin, order, bounds, threads, out);

  if (threads > 1)
    return offline([&] {
      checkIndices(m_alive, indices);
      runParallel(indices.size(), m_cells.size(), threads,
                  [&] (const size_t i) { return objectCell(indices[i]); },
                  [&] { return [&] (const size_t i) { out[i] = countObjects(indices[i], Rmax, Rmin); }; });
    });

  runQueries(indices.size(), m_cells.size(),
             [&] (const size_t i) { return objectCell(indices[i]); },
             [&] (const size_t i) { out[i] = countObjects(indices[i], Rmax, Rmin); });
}


// ===========================================================================


void meshsearch::MeshGrid::countObjects (const std::vector<unsigned int>& indices,
                                         const double Rmax, std::vector<std::uint64_t>& out,
                                         const Workers workers) const
{
  countObjects(indices, Rmax, 0., out, workers);
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
