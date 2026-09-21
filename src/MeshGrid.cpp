#include "meshsearch/MeshGrid.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

namespace {

  const double padding_factor = 1.e-6;

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
  if (!(cellsize > 0.))
    throw Error("cellsize must be positive");

  m_X = X;
  m_Y = Y;
  m_Z = Z;
  m_cellsize = cellsize;

  std::vector<std::vector<double>> temp_lims(3, std::vector<double>(2));

  if (limits.empty()) {
    if (m_X.empty())
      throw Error("an empty set of coordinates requires explicit limits");

    const std::vector<double>* coords[3] = {&m_X, &m_Y, &m_Z};
    for (int i=0; i<3; ++i) {
      const auto range = std::minmax_element(coords[i]->begin(), coords[i]->end());
      const double lo = *range.first;
      const double hi = *range.second;
      const double pad = padding_factor*(hi-lo);
      temp_lims[i][0] = lo-pad;
      temp_lims[i][1] = hi+pad;
    }
  }
  else {
    if (limits.size() != 3 || limits[0].size() != 2 || limits[1].size() != 2 || limits[2].size() != 2)
      throw Error("the limits matrix must be 3x2");

    if (!m_X.empty()) {
      const std::vector<double>* coords[3] = {&m_X, &m_Y, &m_Z};
      const char* names[3] = {"X", "Y", "Z"};
      for (int i=0; i<3; ++i) {
        const auto range = std::minmax_element(coords[i]->begin(), coords[i]->end());
        if (*range.first < limits[i][0] || *range.second > limits[i][1])
          throw Error(std::string(names[i])+" coordinate outside the limits");
      }
    }

    temp_lims = limits;
  }

  for (int i=0; i<3; ++i) {
    const double range = temp_lims[i][1]-temp_lims[i][0];
    const double n = std::ceil(range/m_cellsize);
    m_nCells[i] = (n >= 1.) ? static_cast<unsigned int>(n) : 1;
  }

  m_lims.assign(3, std::vector<double>(2));
  for (int i=0; i<3; ++i) {
    const double range = temp_lims[i][1]-temp_lims[i][0];
    const double grid_span = m_nCells[i]*m_cellsize;
    const double offset = (grid_span-range)/2.;

    m_lims[i][0] = temp_lims[i][0]-offset;
    m_lims[i][1] = temp_lims[i][1]+offset;
  }

  const int nCx = static_cast<int>(m_nCells[0]);
  const int nCy = static_cast<int>(m_nCells[1]);
  const int nCz = static_cast<int>(m_nCells[2]);

  const unsigned int dim_mask = static_cast<unsigned int>
    (std::ceil(std::sqrt(3.)+std::sqrt(double(nCx*nCx+nCy*nCy+nCz*nCz))));

  auto mask = std::make_shared<std::vector<std::vector<GridOffset>>>(dim_mask);

  for (int i=-nCx; i<nCx; ++i) {
    const int abs_i = std::abs(i);
    const int dx = (abs_i > 0) ? abs_i-1 : 0;
    const int dx2 = dx*dx;
    for (int j=-nCy; j<nCy; ++j) {
      const int abs_j = std::abs(j);
      const int dy = (abs_j > 0) ? abs_j-1 : 0;
      const int dy2 = dy*dy;
      for (int k=-nCz; k<nCz; ++k) {
        const int abs_k = std::abs(k);
        const int dz = (abs_k > 0) ? abs_k-1 : 0;
        const unsigned int indexes = static_cast<unsigned int>(std::floor(std::sqrt(double(dx2+dy2+dz*dz))));
        if (indexes < dim_mask) (*mask)[indexes].push_back({i, j, k});
      }
    }
  }

  m_mask = mask;

  const size_t total_cells = size_t(m_nCells[0])*size_t(m_nCells[1])*size_t(m_nCells[2]);
  m_grid.assign(total_cells, std::vector<unsigned int>());

  m_alive.assign(m_X.size(), 1);
  m_nObjects = static_cast<unsigned int>(m_X.size());

  for (size_t i=0; i<m_X.size(); ++i) {
    const auto coords = locate(m_lims, m_nCells, m_cellsize, m_X[i], m_Y[i], m_Z[i]);
    m_grid[getLinearIndex(coords[0], coords[1], coords[2])].push_back(static_cast<unsigned int>(i));
  }
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
      const int currX = targetX+offset.i;
      const int currY = targetY+offset.j;
      const int currZ = targetZ+offset.k;
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

  return finalInd;
}


// ===========================================================================


unsigned int meshsearch::MeshGrid::nearestObject (const unsigned int index) const
{
  checkIndex(m_alive, index);
  if (m_nObjects < 2) throw Error("no other object remains in the grid");

  const double X = m_X[index];
  const double Y = m_Y[index];
  const double Z = m_Z[index];

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
      const int currX = targetX+offset.i;
      const int currY = targetY+offset.j;
      const int currZ = targetZ+offset.k;
      if (currX >= 0 && currX < nCx && currY >= 0 && currY < nCy && currZ >= 0 && currZ < nCz) {
        for (const auto& part_idx : m_grid[getLinearIndex(static_cast<unsigned int>(currX), static_cast<unsigned int>(currY), static_cast<unsigned int>(currZ))]) {
          if (part_idx == index) continue;
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

  return finalInd;
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
      const int currX = targetX+offset.i;
      const int currY = targetY+offset.j;
      const int currZ = targetZ+offset.k;
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
    finalInd[i] = pq.top().index;
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

  const double X = m_X[index];
  const double Y = m_Y[index];
  const double Z = m_Z[index];

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
      const int currX = targetX+offset.i;
      const int currY = targetY+offset.j;
      const int currZ = targetZ+offset.k;
      if (currX >= 0 && currX < nCx && currY >= 0 && currY < nCy && currZ >= 0 && currZ < nCz) {
        for (const auto& part_idx : m_grid[getLinearIndex(static_cast<unsigned int>(currX), static_cast<unsigned int>(currY), static_cast<unsigned int>(currZ))]) {
          if (part_idx == index) continue;
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
    finalInd[i] = pq.top().index;
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

  const double upper = std::ceil(Rmax*inv_cellsize)+3.;
  const unsigned int max_mask_idx = static_cast<unsigned int>((upper < mask_size) ? upper : mask_size);

  const double lower = std::floor(Rmin*inv_cellsize)-4.;
  const unsigned int min_mask_idx = static_cast<unsigned int>((lower < 0.) ? 0. : ((lower < mask_size) ? lower : mask_size));

  std::vector<unsigned int> finalInd;

  for (unsigned int maskInd=min_mask_idx; maskInd<max_mask_idx; maskInd++) {
    if (maskInd > 4) {
      const double min_shell_dist = (maskInd-3.)*m_cellsize;
      if (min_shell_dist*min_shell_dist > Rmax_sq) break;
    }

    for (const auto& offset : (*m_mask)[maskInd]) {
      const int currX = targetX+offset.i;
      const int currY = targetY+offset.j;
      const int currZ = targetZ+offset.k;
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

  return finalInd;
}


// ===========================================================================


std::vector<unsigned int> meshsearch::MeshGrid::closeObjects (const unsigned int index,
                                                              const double Rmax, const double Rmin) const
{
  checkIndex(m_alive, index);
  checkRadii(Rmax, Rmin);

  const double X = m_X[index];
  const double Y = m_Y[index];
  const double Z = m_Z[index];

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

  const double upper = std::ceil(Rmax*inv_cellsize)+3.;
  const unsigned int max_mask_idx = static_cast<unsigned int>((upper < mask_size) ? upper : mask_size);

  const double lower = std::floor(Rmin*inv_cellsize)-4.;
  const unsigned int min_mask_idx = static_cast<unsigned int>((lower < 0.) ? 0. : ((lower < mask_size) ? lower : mask_size));

  std::vector<unsigned int> finalInd;

  for (unsigned int maskInd=min_mask_idx; maskInd<max_mask_idx; maskInd++) {
    if (maskInd > 4) {
      const double min_shell_dist = (maskInd-3.)*m_cellsize;
      if (min_shell_dist*min_shell_dist > Rmax_sq) break;
    }

    for (const auto& offset : (*m_mask)[maskInd]) {
      const int currX = targetX+offset.i;
      const int currY = targetY+offset.j;
      const int currZ = targetZ+offset.k;
      if (currX >= 0 && currX < nCx && currY >= 0 && currY < nCy && currZ >= 0 && currZ < nCz) {
        for (const auto& part_idx : m_grid[getLinearIndex(static_cast<unsigned int>(currX), static_cast<unsigned int>(currY), static_cast<unsigned int>(currZ))]) {
          if (part_idx == index) continue;
          const double dx = m_X[part_idx]-X;
          const double dy = m_Y[part_idx]-Y;
          const double dz = m_Z[part_idx]-Z;
          const double dist_sq = dx*dx+dy*dy+dz*dz;
          if (dist_sq <= Rmax_sq && dist_sq >= Rmin_sq) finalInd.push_back(part_idx);
        }
      }
    }
  }

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
  return m_grid[getLinearIndex(coords[0], coords[1], coords[2])];
}


// ===========================================================================


void meshsearch::MeshGrid::removeObject (const unsigned int index)
{
  checkIndex(m_alive, index);

  const auto coords = locate(m_lims, m_nCells, m_cellsize, m_X[index], m_Y[index], m_Z[index]);
  auto& cell = m_grid[getLinearIndex(coords[0], coords[1], coords[2])];
  cell.erase(std::remove(cell.begin(), cell.end(), index), cell.end());

  m_alive[index] = 0;
  m_nObjects--;
}


// ===========================================================================


unsigned int meshsearch::MeshGrid::addObject (const double X, const double Y, const double Z)
{
  const auto coords = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);

  const unsigned int index = static_cast<unsigned int>(m_X.size());

  m_X.push_back(X);
  m_Y.push_back(Y);
  m_Z.push_back(Z);
  m_alive.push_back(1);
  m_nObjects++;

  m_grid[getLinearIndex(coords[0], coords[1], coords[2])].push_back(index);

  return index;
}
