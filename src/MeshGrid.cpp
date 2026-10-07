#include "meshsearch/MeshGrid.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <string>

namespace {

  const double padding_factor = 1.e-6;

  const char* const axis_names[3] = {"X", "Y", "Z"};

  const unsigned int max_objects = std::numeric_limits<unsigned int>::max();

  bool checkedProduct (const std::uint64_t a, const std::uint64_t b, std::uint64_t& out)
  {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max()/a) return false;
    out = a*b;
    return true;
  }

  struct PartNode {
    double dist_sq;
    unsigned int index;
    bool operator< (const PartNode& other) const { return dist_sq < other.dist_sq; }
  };

  std::array<unsigned int, 3> locate (const std::vector<std::vector<double>>& lims,
                                      const std::array<unsigned int, 3>& nCells,
                                      const double cellsize,
                                      const double X, const double Y, const double Z)
  {
    if (lims.size() != 3) throw meshsearch::Error("the grid holds no box");
    if (!(X >= lims[0][0]) || !(X <= lims[0][1])) throw meshsearch::Error("X coordinate outside the box");
    if (!(Y >= lims[1][0]) || !(Y <= lims[1][1])) throw meshsearch::Error("Y coordinate outside the box");
    if (!(Z >= lims[2][0]) || !(Z <= lims[2][1])) throw meshsearch::Error("Z coordinate outside the box");

    const double inv_cellsize = 1./cellsize;
    const double pos[3] = {(X-lims[0][0])*inv_cellsize,
                           (Y-lims[1][0])*inv_cellsize,
                           (Z-lims[2][0])*inv_cellsize};

    std::array<unsigned int, 3> coords = {0, 0, 0};
    for (int i=0; i<3; ++i) {
      const double floored = std::floor(pos[i]);
      unsigned int ind = (floored > 0.) ? static_cast<unsigned int>(floored) : 0;
      if (ind >= nCells[i]) ind = nCells[i]-1;
      coords[i] = ind;
    }

    return coords;
  }

  void checkIndex (const std::vector<char>& alive, const unsigned int index)
  {
    if (index >= alive.size()) throw meshsearch::IndexError("object index out of range");
    if (alive[index] == 0) throw meshsearch::IndexError("object index has been removed");
  }

  void checkRadii (const double Rmax, const double Rmin)
  {
    if (std::isnan(Rmax) || std::isnan(Rmin)) throw meshsearch::Error("a radius is NaN");
    if (Rmax < 0.) throw meshsearch::Error("Rmax < 0");
    if (Rmin < 0.) throw meshsearch::Error("Rmin < 0");
    if (Rmax < Rmin) throw meshsearch::Error("Rmax < Rmin");
  }

}


// ===========================================================================


meshsearch::Error::Error (const std::string& what)
  : std::runtime_error(what) {}


// ===========================================================================


meshsearch::IndexError::IndexError (const std::string& what)
  : Error(what) {}


// ===========================================================================


meshsearch::MeshGrid::MeshGrid (const std::vector<double>& X,
                                const std::vector<double>& Y,
                                const std::vector<double>& Z,
                                const double cellsize,
                                const std::vector<std::vector<double>>& limits)
{
  if (X.size() != Y.size() || X.size() != Z.size())
    throw Error("the coordinate vectors differ in length");
  if (!(std::isfinite(cellsize) && cellsize > 0.))
    throw Error("cellsize must be positive and finite");
  if (X.size() > max_objects)
    throw Error("too many objects: at most 4294967295 are supported");

  const std::vector<double>* axes[3] = {&X, &Y, &Z};
  for (int i=0; i<3; ++i)
    for (size_t k=0; k<axes[i]->size(); ++k)
      if (!std::isfinite((*axes[i])[k]))
        throw Error(std::string(axis_names[i])+" coordinate "+std::to_string(k)+" is not finite");

  double lo[3], hi[3];

  if (limits.empty()) {
    if (X.empty())
      throw Error("an empty set of coordinates requires explicit limits");

    for (int i=0; i<3; ++i) {
      const auto range = std::minmax_element(axes[i]->begin(), axes[i]->end());
      const double pad = padding_factor*(*range.second-*range.first);
      lo[i] = *range.first-pad;
      hi[i] = *range.second+pad;
    }
  }
  else {
    if (limits.size() != 3 || limits[0].size() != 2 || limits[1].size() != 2 || limits[2].size() != 2)
      throw Error("the limits matrix must be 3x2");

    for (int i=0; i<3; ++i) {
      if (!std::isfinite(limits[i][0]) || !std::isfinite(limits[i][1]))
        throw Error("the limits must be finite");
      if (limits[i][0] > limits[i][1])
        throw Error(std::string(axis_names[i])+" limits are inverted (lo > hi)");
    }

    if (!X.empty()) {
      for (int i=0; i<3; ++i) {
        const auto range = std::minmax_element(axes[i]->begin(), axes[i]->end());
        if (*range.first < limits[i][0] || *range.second > limits[i][1])
          throw Error(std::string(axis_names[i])+" coordinate outside the limits");
      }
    }

    for (int i=0; i<3; ++i) {
      lo[i] = limits[i][0];
      hi[i] = limits[i][1];
    }
  }

  std::array<unsigned int, 3> nCells;
  for (int i=0; i<3; ++i) {
    const double n = std::ceil((hi[i]-lo[i])/cellsize);
    if (!(n <= double(std::numeric_limits<int>::max())))
      throw Error("the grid would need more than 2147483647 cells along "+std::string(axis_names[i]));
    nCells[i] = (n >= 1.) ? static_cast<unsigned int>(n) : 1;
  }

  double grownLo[3], grownHi[3];
  for (int i=0; i<3; ++i) {
    const double range = hi[i]-lo[i];
    const double grid_span = nCells[i]*cellsize;
    const double offset = (grid_span-range)/2.;

    grownLo[i] = lo[i]-offset;
    grownHi[i] = hi[i]+offset;
    if (!std::isfinite(grownLo[i]) || !std::isfinite(grownHi[i]))
      throw Error("the box is too large to represent");
  }

  const std::uint64_t nx = nCells[0], ny = nCells[1], nz = nCells[2];
  const char* const too_many_cells = "the grid would need more cells than 64-bit sizes can count or address";

  std::uint64_t total = 0, entries = 0;
  if (!checkedProduct(nx, ny, total) || !checkedProduct(total, nz, total) || !checkedProduct(total, 8, entries))
    throw Error(too_many_cells);
  if (total >= std::vector<size_t>().max_size() || total > std::vector<std::vector<unsigned int>>().max_size()
      || entries > std::vector<GridOffset>().max_size())
    throw Error(too_many_cells);

  m_cellsize = cellsize;
  m_nCells = nCells;
  m_lims = {{grownLo[0], grownHi[0]}, {grownLo[1], grownHi[1]}, {grownLo[2], grownHi[2]}};

  const int nCx = static_cast<int>(m_nCells[0]);
  const int nCy = static_cast<int>(m_nCells[1]);
  const int nCz = static_cast<int>(m_nCells[2]);

  const std::uint64_t sum_sq = nx*nx+ny*ny+nz*nz;
  const unsigned int dim_mask = static_cast<unsigned int>
    (std::ceil(std::sqrt(3.)+std::sqrt(double(sum_sq))));

  auto mask = std::make_shared<std::vector<std::vector<GridOffset>>>(dim_mask);

  for (int i=-nCx; i<nCx; ++i) {
    const int abs_i = std::abs(i);
    const std::uint64_t dx = (abs_i > 0) ? abs_i-1 : 0;
    const std::uint64_t dx2 = dx*dx;
    for (int j=-nCy; j<nCy; ++j) {
      const int abs_j = std::abs(j);
      const std::uint64_t dy = (abs_j > 0) ? abs_j-1 : 0;
      const std::uint64_t dy2 = dy*dy;
      for (int k=-nCz; k<nCz; ++k) {
        const int abs_k = std::abs(k);
        const std::uint64_t dz = (abs_k > 0) ? abs_k-1 : 0;
        const std::uint64_t sum = dx2+dy2+dz*dz;
        std::uint64_t root = static_cast<std::uint64_t>(std::floor(std::sqrt(double(sum))));
        while (root*root > sum) --root;
        while ((root+1)*(root+1) <= sum) ++root;
        if (root < dim_mask) (*mask)[static_cast<size_t>(root)].push_back({i, j, k});
      }
    }
  }

  m_mask = mask;

  const size_t total_cells = static_cast<size_t>(total);
  m_grid.assign(total_cells, std::vector<unsigned int>());

  const size_t n = X.size();

  std::vector<size_t> cell(n);
  std::vector<size_t> offset(total_cells+1, 0);

  for (size_t i=0; i<n; ++i) {
    const auto coords = locate(m_lims, m_nCells, m_cellsize, X[i], Y[i], Z[i]);
    cell[i] = getLinearIndex(coords[0], coords[1], coords[2]);
    offset[cell[i]+1]++;
  }

  for (size_t c=0; c<total_cells; ++c) offset[c+1] += offset[c];

  m_toPublic.resize(n);
  m_toInternal.resize(n);

  for (size_t i=0; i<n; ++i) {
    const size_t internal = offset[cell[i]]++;
    m_toPublic[internal] = static_cast<unsigned int>(i);
    m_toInternal[i] = static_cast<unsigned int>(internal);
  }

  m_X.resize(n);
  m_Y.resize(n);
  m_Z.resize(n);

  for (size_t k=0; k<n; ++k) {
    const unsigned int index = m_toPublic[k];
    m_X[k] = X[index];
    m_Y[k] = Y[index];
    m_Z[k] = Z[index];
  }

  size_t start = 0;
  for (size_t c=0; c<total_cells; ++c) {
    const size_t end = offset[c];
    m_grid[c].reserve(end-start);
    for (size_t k=start; k<end; ++k) m_grid[c].push_back(static_cast<unsigned int>(k));
    start = end;
  }

  m_alive.assign(n, 1);
  m_nObjects = static_cast<unsigned int>(n);
}


// ===========================================================================


size_t meshsearch::MeshGrid::getLinearIndex (const unsigned int ix, const unsigned int iy, const unsigned int iz) const
{
  return size_t(ix)*size_t(m_nCells[1])*size_t(m_nCells[2])+size_t(iy)*size_t(m_nCells[2])+size_t(iz);
}


// ===========================================================================


unsigned int meshsearch::MeshGrid::get_nObjects () const
{
  return m_nObjects;
}


// ===========================================================================


bool meshsearch::MeshGrid::isAlive (const unsigned int index) const
{
  return index < m_alive.size() && m_alive[index] != 0;
}


// ===========================================================================


std::vector<std::vector<double>> meshsearch::MeshGrid::get_lims () const
{
  return m_lims;
}


// ===========================================================================


std::array<unsigned int, 3> meshsearch::MeshGrid::get_nCells () const
{
  return m_nCells;
}


// ===========================================================================


double meshsearch::MeshGrid::get_cellsize () const
{
  return m_cellsize;
}


// ===========================================================================


unsigned int meshsearch::MeshGrid::nearestObject (const double X, const double Y, const double Z) const
{
  const auto target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  if (m_nObjects == 0) throw Error("the grid holds no objects");

  const int targetX = static_cast<int>(target[0]);
  const int targetY = static_cast<int>(target[1]);
  const int targetZ = static_cast<int>(target[2]);

  const int nCx = static_cast<int>(m_nCells[0]);
  const int nCy = static_cast<int>(m_nCells[1]);
  const int nCz = static_cast<int>(m_nCells[2]);

  double min_dist_sq = std::numeric_limits<double>::max();
  unsigned int finalInd = std::numeric_limits<unsigned int>::max();
  unsigned int maskInd = 0;

  while (maskInd < m_mask->size()) {
    const double layer_dist = maskInd*m_cellsize;
    if (layer_dist*layer_dist >= min_dist_sq) break;

    for (const auto& offset : (*m_mask)[maskInd]) {
      const std::int64_t currX = std::int64_t(targetX)+offset.i;
      const std::int64_t currY = std::int64_t(targetY)+offset.j;
      const std::int64_t currZ = std::int64_t(targetZ)+offset.k;
      if (currX >= 0 && currX < nCx && currY >= 0 && currY < nCy && currZ >= 0 && currZ < nCz) {
        for (const auto& part_idx : m_grid[getLinearIndex(static_cast<unsigned int>(currX), static_cast<unsigned int>(currY), static_cast<unsigned int>(currZ))]) {
          const double dx = m_X[part_idx]-X;
          const double dy = m_Y[part_idx]-Y;
          const double dz = m_Z[part_idx]-Z;
          const double dist_sq_temp = dx*dx+dy*dy+dz*dz;
          if (dist_sq_temp < min_dist_sq) {
            min_dist_sq = dist_sq_temp;
            finalInd = part_idx;
          }
        }
      }
    }
    maskInd++;
  }

  if (finalInd == std::numeric_limits<unsigned int>::max())
    throw Error("no object found");

  return m_toPublic[finalInd];
}


// ===========================================================================


unsigned int meshsearch::MeshGrid::nearestObject (const unsigned int index) const
{
  checkIndex(m_alive, index);
  if (m_nObjects < 2) throw Error("no other object remains in the grid");

  const unsigned int self = m_toInternal[index];
  const double X = m_X[self];
  const double Y = m_Y[self];
  const double Z = m_Z[self];

  const auto target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  const int targetX = static_cast<int>(target[0]);
  const int targetY = static_cast<int>(target[1]);
  const int targetZ = static_cast<int>(target[2]);

  const int nCx = static_cast<int>(m_nCells[0]);
  const int nCy = static_cast<int>(m_nCells[1]);
  const int nCz = static_cast<int>(m_nCells[2]);

  double min_dist_sq = std::numeric_limits<double>::max();
  unsigned int finalInd = std::numeric_limits<unsigned int>::max();
  unsigned int maskInd = 0;

  while (maskInd < m_mask->size()) {
    const double layer_dist = maskInd*m_cellsize;
    if (layer_dist*layer_dist >= min_dist_sq) break;

    for (const auto& offset : (*m_mask)[maskInd]) {
      const std::int64_t currX = std::int64_t(targetX)+offset.i;
      const std::int64_t currY = std::int64_t(targetY)+offset.j;
      const std::int64_t currZ = std::int64_t(targetZ)+offset.k;
      if (currX >= 0 && currX < nCx && currY >= 0 && currY < nCy && currZ >= 0 && currZ < nCz) {
        for (const auto& part_idx : m_grid[getLinearIndex(static_cast<unsigned int>(currX), static_cast<unsigned int>(currY), static_cast<unsigned int>(currZ))]) {
          if (part_idx == self) continue;
          const double dx = m_X[part_idx]-X;
          const double dy = m_Y[part_idx]-Y;
          const double dz = m_Z[part_idx]-Z;
          const double dist_sq_temp = dx*dx+dy*dy+dz*dz;
          if (dist_sq_temp < min_dist_sq) {
            min_dist_sq = dist_sq_temp;
            finalInd = part_idx;
          }
        }
      }
    }
    maskInd++;
  }

  if (finalInd == std::numeric_limits<unsigned int>::max())
    throw Error("no object found");

  return m_toPublic[finalInd];
}


// ===========================================================================


std::vector<unsigned int> meshsearch::MeshGrid::nearestObjects (const unsigned int N,
                                                                const double X, const double Y, const double Z) const
{
  const auto target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  if (N > m_nObjects) throw Error("N is larger than the number of objects");
  if (N == 0) return {};

  const int targetX = static_cast<int>(target[0]);
  const int targetY = static_cast<int>(target[1]);
  const int targetZ = static_cast<int>(target[2]);

  const int nCx = static_cast<int>(m_nCells[0]);
  const int nCy = static_cast<int>(m_nCells[1]);
  const int nCz = static_cast<int>(m_nCells[2]);

  std::priority_queue<PartNode> pq;
  unsigned int maskInd = 0;

  while (maskInd < m_mask->size()) {
    if (pq.size() == N) {
      const double min_shell_dist = (maskInd > 1) ? (maskInd-1.)*m_cellsize : 0.;
      if (min_shell_dist*min_shell_dist > pq.top().dist_sq) break;
    }

    for (const auto& offset : (*m_mask)[maskInd]) {
      const std::int64_t currX = std::int64_t(targetX)+offset.i;
      const std::int64_t currY = std::int64_t(targetY)+offset.j;
      const std::int64_t currZ = std::int64_t(targetZ)+offset.k;
      if (currX >= 0 && currX < nCx && currY >= 0 && currY < nCy && currZ >= 0 && currZ < nCz) {
        for (const auto& part_idx : m_grid[getLinearIndex(static_cast<unsigned int>(currX), static_cast<unsigned int>(currY), static_cast<unsigned int>(currZ))]) {
          const double dx = m_X[part_idx]-X;
          const double dy = m_Y[part_idx]-Y;
          const double dz = m_Z[part_idx]-Z;
          const double dist_sq = dx*dx+dy*dy+dz*dz;
          if (pq.size() < N) pq.push({dist_sq, part_idx});
          else if (dist_sq < pq.top().dist_sq) {
            pq.pop();
            pq.push({dist_sq, part_idx});
          }
        }
      }
    }
    maskInd++;
  }

  std::vector<unsigned int> finalInd(pq.size());
  for (size_t i=finalInd.size(); i-- > 0; ) {
    finalInd[i] = m_toPublic[pq.top().index];
    pq.pop();
  }

  return finalInd;
}


// ===========================================================================


std::vector<unsigned int> meshsearch::MeshGrid::nearestObjects (const unsigned int N,
                                                                const unsigned int index) const
{
  checkIndex(m_alive, index);
  if (N >= m_nObjects) throw Error("N is larger than the number of available neighbours");
  if (N == 0) return {};

  const unsigned int self = m_toInternal[index];
  const double X = m_X[self];
  const double Y = m_Y[self];
  const double Z = m_Z[self];

  const auto target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  const int targetX = static_cast<int>(target[0]);
  const int targetY = static_cast<int>(target[1]);
  const int targetZ = static_cast<int>(target[2]);

  const int nCx = static_cast<int>(m_nCells[0]);
  const int nCy = static_cast<int>(m_nCells[1]);
  const int nCz = static_cast<int>(m_nCells[2]);

  std::priority_queue<PartNode> pq;
  unsigned int maskInd = 0;

  while (maskInd < m_mask->size()) {
    if (pq.size() == N) {
      const double min_shell_dist = (maskInd > 1) ? (maskInd-1.)*m_cellsize : 0.;
      if (min_shell_dist*min_shell_dist > pq.top().dist_sq) break;
    }

    for (const auto& offset : (*m_mask)[maskInd]) {
      const std::int64_t currX = std::int64_t(targetX)+offset.i;
      const std::int64_t currY = std::int64_t(targetY)+offset.j;
      const std::int64_t currZ = std::int64_t(targetZ)+offset.k;
      if (currX >= 0 && currX < nCx && currY >= 0 && currY < nCy && currZ >= 0 && currZ < nCz) {
        for (const auto& part_idx : m_grid[getLinearIndex(static_cast<unsigned int>(currX), static_cast<unsigned int>(currY), static_cast<unsigned int>(currZ))]) {
          if (part_idx == self) continue;
          const double dx = m_X[part_idx]-X;
          const double dy = m_Y[part_idx]-Y;
          const double dz = m_Z[part_idx]-Z;
          const double dist_sq = dx*dx+dy*dy+dz*dz;
          if (pq.size() < N) pq.push({dist_sq, part_idx});
          else if (dist_sq < pq.top().dist_sq) {
            pq.pop();
            pq.push({dist_sq, part_idx});
          }
        }
      }
    }
    maskInd++;
  }

  std::vector<unsigned int> finalInd(pq.size());
  for (size_t i=finalInd.size(); i-- > 0; ) {
    finalInd[i] = m_toPublic[pq.top().index];
    pq.pop();
  }

  return finalInd;
}


// ===========================================================================


std::vector<unsigned int> meshsearch::MeshGrid::closeObjects (const double X, const double Y, const double Z,
                                                              const double Rmax, const double Rmin) const
{
  const auto target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  checkRadii(Rmax, Rmin);

  const double Rmax_sq = Rmax*Rmax;
  const double Rmin_sq = Rmin*Rmin;

  const double inv_cellsize = 1./m_cellsize;

  const int targetX = static_cast<int>(target[0]);
  const int targetY = static_cast<int>(target[1]);
  const int targetZ = static_cast<int>(target[2]);

  const int nCx = static_cast<int>(m_nCells[0]);
  const int nCy = static_cast<int>(m_nCells[1]);
  const int nCz = static_cast<int>(m_nCells[2]);

  const double mask_size = static_cast<double>(m_mask->size());

  const double upper = std::floor(Rmax/m_cellsize)+1.;
  const unsigned int max_mask_idx = static_cast<unsigned int>((upper < mask_size) ? upper : mask_size);

  const double lower = std::floor(Rmin*inv_cellsize)-4.;
  const unsigned int min_mask_idx = static_cast<unsigned int>((lower < 0.) ? 0. : ((lower < mask_size) ? lower : mask_size));

  std::vector<unsigned int> finalInd;

  for (unsigned int maskInd=min_mask_idx; maskInd<max_mask_idx; maskInd++) {
    const double layer_dist = maskInd*m_cellsize;
    if (layer_dist*layer_dist > Rmax_sq) break;

    for (const auto& offset : (*m_mask)[maskInd]) {
      const std::int64_t currX = std::int64_t(targetX)+offset.i;
      const std::int64_t currY = std::int64_t(targetY)+offset.j;
      const std::int64_t currZ = std::int64_t(targetZ)+offset.k;
      if (currX >= 0 && currX < nCx && currY >= 0 && currY < nCy && currZ >= 0 && currZ < nCz) {
        for (const auto& part_idx : m_grid[getLinearIndex(static_cast<unsigned int>(currX), static_cast<unsigned int>(currY), static_cast<unsigned int>(currZ))]) {
          const double dx = m_X[part_idx]-X;
          const double dy = m_Y[part_idx]-Y;
          const double dz = m_Z[part_idx]-Z;
          const double dist_sq = dx*dx+dy*dy+dz*dz;
          if (dist_sq <= Rmax_sq && dist_sq >= Rmin_sq) finalInd.push_back(part_idx);
        }
      }
    }
  }

  for (auto& part_idx : finalInd) part_idx = m_toPublic[part_idx];

  return finalInd;
}


// ===========================================================================


std::vector<unsigned int> meshsearch::MeshGrid::closeObjects (const unsigned int index,
                                                              const double Rmax, const double Rmin) const
{
  checkIndex(m_alive, index);
  checkRadii(Rmax, Rmin);

  const unsigned int self = m_toInternal[index];
  const double X = m_X[self];
  const double Y = m_Y[self];
  const double Z = m_Z[self];

  const double Rmax_sq = Rmax*Rmax;
  const double Rmin_sq = Rmin*Rmin;

  const double inv_cellsize = 1./m_cellsize;

  const auto target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  const int targetX = static_cast<int>(target[0]);
  const int targetY = static_cast<int>(target[1]);
  const int targetZ = static_cast<int>(target[2]);

  const int nCx = static_cast<int>(m_nCells[0]);
  const int nCy = static_cast<int>(m_nCells[1]);
  const int nCz = static_cast<int>(m_nCells[2]);

  const double mask_size = static_cast<double>(m_mask->size());

  const double upper = std::floor(Rmax/m_cellsize)+1.;
  const unsigned int max_mask_idx = static_cast<unsigned int>((upper < mask_size) ? upper : mask_size);

  const double lower = std::floor(Rmin*inv_cellsize)-4.;
  const unsigned int min_mask_idx = static_cast<unsigned int>((lower < 0.) ? 0. : ((lower < mask_size) ? lower : mask_size));

  std::vector<unsigned int> finalInd;

  for (unsigned int maskInd=min_mask_idx; maskInd<max_mask_idx; maskInd++) {
    const double layer_dist = maskInd*m_cellsize;
    if (layer_dist*layer_dist > Rmax_sq) break;

    for (const auto& offset : (*m_mask)[maskInd]) {
      const std::int64_t currX = std::int64_t(targetX)+offset.i;
      const std::int64_t currY = std::int64_t(targetY)+offset.j;
      const std::int64_t currZ = std::int64_t(targetZ)+offset.k;
      if (currX >= 0 && currX < nCx && currY >= 0 && currY < nCy && currZ >= 0 && currZ < nCz) {
        for (const auto& part_idx : m_grid[getLinearIndex(static_cast<unsigned int>(currX), static_cast<unsigned int>(currY), static_cast<unsigned int>(currZ))]) {
          if (part_idx == self) continue;
          const double dx = m_X[part_idx]-X;
          const double dy = m_Y[part_idx]-Y;
          const double dz = m_Z[part_idx]-Z;
          const double dist_sq = dx*dx+dy*dy+dz*dz;
          if (dist_sq <= Rmax_sq && dist_sq >= Rmin_sq) finalInd.push_back(part_idx);
        }
      }
    }
  }

  for (auto& part_idx : finalInd) part_idx = m_toPublic[part_idx];

  return finalInd;
}


// ===========================================================================


size_t meshsearch::MeshGrid::get_Cell (const double X, const double Y, const double Z) const
{
  const auto coords = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  return getLinearIndex(coords[0], coords[1], coords[2]);
}


// ===========================================================================


std::array<unsigned int, 3> meshsearch::MeshGrid::get_CellCoords (const double X, const double Y, const double Z) const
{
  return locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
}


// ===========================================================================


std::vector<unsigned int> meshsearch::MeshGrid::get_ObjectsInCell (const double X, const double Y, const double Z) const
{
  const auto coords = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  std::vector<unsigned int> objects = m_grid[getLinearIndex(coords[0], coords[1], coords[2])];

  for (auto& part_idx : objects) part_idx = m_toPublic[part_idx];

  return objects;
}


// ===========================================================================


void meshsearch::MeshGrid::removeObject (const unsigned int index)
{
  checkIndex(m_alive, index);

  const unsigned int internal = m_toInternal[index];

  const auto coords = locate(m_lims, m_nCells, m_cellsize, m_X[internal], m_Y[internal], m_Z[internal]);
  auto& cell = m_grid[getLinearIndex(coords[0], coords[1], coords[2])];
  cell.erase(std::remove(cell.begin(), cell.end(), internal), cell.end());

  m_alive[index] = 0;
  m_nObjects--;
}


// ===========================================================================


unsigned int meshsearch::MeshGrid::addObject (const double X, const double Y, const double Z)
{
  const auto coords = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  if (m_toInternal.size() >= max_objects)
    throw Error("too many objects: at most 4294967295 indices can be issued");

  const unsigned int index = static_cast<unsigned int>(m_toInternal.size());
  const unsigned int internal = static_cast<unsigned int>(m_X.size());

  auto& cell = m_grid[getLinearIndex(coords[0], coords[1], coords[2])];

  int pushed = 0;
  try {
    m_X.push_back(X); ++pushed;
    m_Y.push_back(Y); ++pushed;
    m_Z.push_back(Z); ++pushed;
    m_toInternal.push_back(internal); ++pushed;
    m_toPublic.push_back(index); ++pushed;
    m_alive.push_back(1); ++pushed;
    cell.push_back(internal);
  }
  catch (...) {
    if (pushed > 5) m_alive.pop_back();
    if (pushed > 4) m_toPublic.pop_back();
    if (pushed > 3) m_toInternal.pop_back();
    if (pushed > 2) m_Z.pop_back();
    if (pushed > 1) m_Y.pop_back();
    if (pushed > 0) m_X.pop_back();
    throw;
  }

  m_nObjects++;

  return index;
}
