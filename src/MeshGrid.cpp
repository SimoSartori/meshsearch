#include "meshsearch/MeshGrid.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace {

  const double padding_factor = 1.e-6;

  const char* const axis_names[3] = {"X", "Y", "Z"};

  const unsigned int max_objects = std::numeric_limits<unsigned int>::max();

  const double tiny_sq = 0x1p-1000;

  const double huge_sq = 0x1p1000;

  const double stop_factor = 1.-0x1p-14;

  const double pad_scale = 0x1p-49;

  const double pad_factor = 1.+0x1p-40;

  const double skip_factor = 1.-0x1p-40;

  const double far_reach = 3.46413;

  const double periodic_tolerance = 1.e-12;

  const double periodic_slack = 0x1.4p-49;

  const double plain_slack = 0x1.ap-50;

  const size_t scan_chunk = 64;

  const unsigned int min_block = 4;

  const size_t no_slot = std::numeric_limits<size_t>::max();

  template <class Visit>
  void scanBlock (const double* x, const double* y, const double* z, const size_t n,
                  const double X, const double Y, const double Z, Visit&& visit)
  {
    double dist_sq[scan_chunk];
    for (size_t base=0; base<n; base+=scan_chunk) {
      const size_t m = std::min(scan_chunk, n-base);
      const double* xb = x+base;
      const double* yb = y+base;
      const double* zb = z+base;
      for (size_t k=0; k<m; ++k) {
        const double dx = xb[k]-X;
        const double dy = yb[k]-Y;
        const double dz = zb[k]-Z;
        dist_sq[k] = dx*dx+dy*dy+dz*dz;
      }
      for (size_t k=0; k<m; ++k) visit(base+k, dist_sq[k]);
    }
  }

  template <class Visit>
  void scanShifted (const double* x, const double* y, const double* z, const size_t n,
                    const double X, const double Y, const double Z,
                    const double sx, const double sy, const double sz, Visit&& visit)
  {
    double dist_sq[scan_chunk];
    for (size_t base=0; base<n; base+=scan_chunk) {
      const size_t m = std::min(scan_chunk, n-base);
      const double* xb = x+base;
      const double* yb = y+base;
      const double* zb = z+base;
      for (size_t k=0; k<m; ++k) {
        const double dx = (xb[k]-X)-sx;
        const double dy = (yb[k]-Y)-sy;
        const double dz = (zb[k]-Z)-sz;
        dist_sq[k] = dx*dx+dy*dy+dz*dz;
      }
      for (size_t k=0; k<m; ++k) visit(base+k, dist_sq[k]);
    }
  }

  double minimumImage (const double d, const double period, const double half)
  {
    if (d >= half) return d-period;
    if (d < -half) return d+period;
    return d;
  }

  template <class Visit>
  void scanImages (const double* x, const double* y, const double* z, const size_t n,
                   const double X, const double Y, const double Z,
                   const double* period, const double* half, Visit&& visit)
  {
    for (size_t k=0; k<n; ++k) {
      const double dx = minimumImage(x[k]-X, period[0], half[0]);
      const double dy = minimumImage(y[k]-Y, period[1], half[1]);
      const double dz = minimumImage(z[k]-Z, period[2], half[2]);
      visit(k, dx*dx+dy*dy+dz*dz);
    }
  }

  bool checkedProduct (const std::uint64_t a, const std::uint64_t b, std::uint64_t& out)
  {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max()/a) return false;
    out = a*b;
    return true;
  }

  struct PartNode {
    double dist_sq;
    unsigned int index;
    bool operator< (const PartNode& other) const
    {
      return dist_sq < other.dist_sq || (dist_sq == other.dist_sq && index < other.index);
    }
  };

  const unsigned int sorted_limit = 56;

  const double no_threshold = std::numeric_limits<double>::infinity();

  class SortedKeep {
  public:
    double T = no_threshold;
    explicit SortedKeep (const unsigned int N) : m_N(N) { m_a.reserve(N); }
    void offer (const double d, const unsigned int idx)
    {
      const PartNode node = {d, idx};
      size_t pos = m_a.size();
      if (pos < m_N) m_a.push_back(node);
      else {
        if (!(node < m_a[pos-1])) return;
        --pos;
      }
      PartNode* a = m_a.data();
      while (pos > 0 && node < a[pos-1]) { a[pos] = a[pos-1]; --pos; }
      a[pos] = node;
      if (m_a.size() == m_N) T = a[m_N-1].dist_sq;
    }
    void finish (std::vector<unsigned int>& out) const
    {
      out.resize(m_a.size());
      for (size_t i=0; i<m_a.size(); ++i) out[i] = m_a[i].index;
    }
  private:
    std::vector<PartNode> m_a;
    size_t m_N;
  };

  class BufferKeep {
  public:
    double T = no_threshold;
    explicit BufferKeep (const unsigned int N) : m_N(N), m_cap(2*size_t(N)) { m_b.reserve(m_cap); }
    void offer (const double d, const unsigned int idx)
    {
      m_b.push_back({d, idx});
      const size_t n = m_b.size();
      if (n == m_N && T == no_threshold) {
        double mx = m_b[0].dist_sq;
        for (size_t i=1; i<n; ++i) mx = (m_b[i].dist_sq > mx) ? m_b[i].dist_sq : mx;
        T = mx;
      }
      else if (n == m_cap) prune();
    }
    void finish (std::vector<unsigned int>& out)
    {
      if (m_b.size() > m_N) std::nth_element(m_b.begin(), m_b.begin()+(m_N-1), m_b.end());
      const size_t n = std::min(m_b.size(), m_N);
      std::sort(m_b.begin(), m_b.begin()+n);
      out.resize(n);
      for (size_t i=0; i<n; ++i) out[i] = m_b[i].index;
    }
  private:
    void prune ()
    {
      std::nth_element(m_b.begin(), m_b.begin()+(m_N-1), m_b.end());
      m_b.resize(m_N);
      T = m_b[m_N-1].dist_sq;
    }
    std::vector<PartNode> m_b;
    size_t m_N, m_cap;
  };

  using Lims = std::array<std::array<double,2>,3>;

  bool inBox (const std::array<unsigned int, 3>& nCells, const Lims& lims,
              const double X, const double Y, const double Z)
  {
    return nCells[0] != 0
      && X >= lims[0][0] && X <= lims[0][1]
      && Y >= lims[1][0] && Y <= lims[1][1]
      && Z >= lims[2][0] && Z <= lims[2][1];
  }

  std::array<unsigned int, 3> locate (const Lims& lims,
                                      const std::array<unsigned int, 3>& nCells,
                                      const double cellsize,
                                      const double X, const double Y, const double Z)
  {
    if (nCells[0] == 0) throw meshsearch::Error("the grid holds no box");
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
      if (floored >= double(nCells[i])) coords[i] = nCells[i]-1;
      else if (floored > 0.) coords[i] = static_cast<unsigned int>(floored);
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

  bool exceeds (const double bound_sq, const double threshold)
  {
    return bound_sq >= tiny_sq && threshold <= huge_sq && bound_sq > threshold;
  }

  bool shellBeyond (const unsigned int shell, const double cellsize, const double threshold)
  {
    if (shell == 0) return false;
    const double reach = shell*cellsize;
    return exceeds((reach*reach)*stop_factor, threshold);
  }

  bool shellBeyondPeriodic (const unsigned int shell, const double cellsize, const double slack, const double threshold)
  {
    if (shell == 0) return false;
    const double cells = double(shell);
    if (!(cells > slack)) return false;
    const double reach = (cells-slack)*cellsize;
    return exceeds((reach*reach)*stop_factor, threshold);
  }

  struct CellFaces {
    double lo[3];
    double pad[3];
    double cellsize;
  };

  CellFaces cellFaces (const Lims& lims,
                       const std::array<unsigned int, 3>& nCells,
                       const double cellsize)
  {
    CellFaces faces;
    faces.cellsize = cellsize;
    for (int i=0; i<3; ++i) {
      const double span = double(nCells[i])*cellsize;
      faces.lo[i] = lims[i][0];
      faces.pad[i] = pad_scale*(std::fabs(lims[i][0])+std::fabs(lims[i][1])+2.*span+2.*cellsize);
    }
    return faces;
  }

  CellFaces periodicFaces (const Lims& lims,
                           const std::array<unsigned int, 3>& nCells,
                           const double cellsize,
                           const std::array<bool, 3>& periodic,
                           const std::array<double, 3>& omega)
  {
    CellFaces faces = cellFaces(lims, nCells, cellsize);
    for (int i=0; i<3; ++i) {
      if (!periodic[i]) continue;
      const double span = double(nCells[i])*cellsize;
      faces.pad[i] = pad_scale*(std::fabs(lims[i][0])+std::fabs(lims[i][1])+4.*span+2.*cellsize)+2.*omega[i];
    }
    return faces;
  }

  double axisGap (const CellFaces& faces, const int axis, const std::int64_t cell, const double q)
  {
    const double lower = faces.lo[axis]+double(cell)*faces.cellsize;
    const double upper = faces.lo[axis]+double(cell+1)*faces.cellsize;
    const double gap = (q < lower) ? lower-q : (q > upper) ? q-upper : 0.;
    const double padded = gap-faces.pad[axis];
    return (padded > 0.) ? padded : 0.;
  }

  bool cellBeyond (const CellFaces& faces, const std::int64_t cx, const std::int64_t cy, const std::int64_t cz,
                   const double X, const double Y, const double Z, const double threshold)
  {
    if (!(threshold <= huge_sq)) return false;
    const double gx = axisGap(faces, 0, cx, X);
    const double gy = axisGap(faces, 1, cy, Y);
    const double gz = axisGap(faces, 2, cz, Z);
    const double bound_sq = gx*gx+gy*gy+gz*gz;
    return bound_sq >= tiny_sq && bound_sq*skip_factor > threshold;
  }

  unsigned int firstShell (const CellFaces& faces, const size_t nShells, const double Rmin, const double Rmin_sq)
  {
    if (!(Rmin_sq >= tiny_sq)) return 0;

    double pad_sq = 0.;
    for (int i=0; i<3; ++i) {
      const double pad = faces.pad[i]/faces.cellsize;
      pad_sq += pad*pad;
    }
    const double pad_cells = std::sqrt(pad_sq)*pad_factor;

    const double start = std::floor((Rmin*(1./faces.cellsize)-pad_cells)-far_reach);
    if (!(start > 0.)) return 0;
    return (start < double(nShells)) ? static_cast<unsigned int>(start) : static_cast<unsigned int>(nShells);
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
                                const Options& opt)
{
  if (X.size() != Y.size() || X.size() != Z.size())
    throw Error("the coordinate vectors differ in length");
  if (!(std::isfinite(cellsize) && cellsize > 0.))
    throw Error("cellsize must be positive and finite");
  if (!std::isfinite(1./cellsize))
    throw Error("cellsize is too small: its reciprocal overflows a double");
  if (X.size() > max_objects)
    throw Error("too many objects: at most 4294967295 are supported");

  const std::vector<double>* axes[3] = {&X, &Y, &Z};
  for (int i=0; i<3; ++i)
    for (size_t k=0; k<axes[i]->size(); ++k)
      if (!std::isfinite((*axes[i])[k]))
        throw Error(std::string(axis_names[i])+" coordinate "+std::to_string(k)+" is not finite");

  const bool any_periodic = opt.periodic[0] || opt.periodic[1] || opt.periodic[2];

  double lo[3], hi[3];

  if (!opt.limits) {
    if (any_periodic)
      throw Error("periodic axes require explicit limits");
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
    const auto& limits = *opt.limits;

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

  double period[3] = {0., 0., 0.};
  double ratio[3] = {0., 0., 0.};
  double side = cellsize;
  unsigned int cells_per_period = 0;

  if (any_periodic) {
    double shortest = std::numeric_limits<double>::infinity();
    for (int i=0; i<3; ++i) {
      if (!opt.periodic[i]) continue;
      period[i] = hi[i]-lo[i];
      if (!(period[i] >= tiny_sq))
        throw Error(std::string(axis_names[i])+" is periodic but its period is below 2^-1000");
      if (!(std::fabs(lo[i])+std::fabs(hi[i])+4.*period[i] <= huge_sq))
        throw Error(std::string(axis_names[i])+" is periodic but its limits are too large for the period to be shifted");
      shortest = std::min(shortest, period[i]);
    }

    for (int i=0; i<3; ++i) {
      if (!opt.periodic[i]) continue;
      ratio[i] = std::nearbyint(period[i]/shortest);
      if (!(ratio[i] >= 1.) || !(std::fabs(period[i]-ratio[i]*shortest) <= periodic_tolerance*period[i]))
        throw Error("the periodic sides must be integer multiples of the shortest one");
    }

    double n = std::ceil(shortest/cellsize);
    if (!(n >= 1.)) n = 1.;
    if (!(n <= double(std::numeric_limits<int>::max())))
      throw Error("the grid would need more than 2147483647 cells along a periodic axis");
    side = shortest/n;
    if (side > cellsize) {
      n += 1.;
      if (!(n <= double(std::numeric_limits<int>::max())))
        throw Error("the grid would need more than 2147483647 cells along a periodic axis");
      side = shortest/n;
    }
    if (!std::isfinite(1./side))
      throw Error("the cell side of a periodic grid is too small: its reciprocal overflows a double");
    cells_per_period = static_cast<unsigned int>(n);
  }

  std::array<unsigned int, 3> nCells;
  for (int i=0; i<3; ++i) {
    if (opt.periodic[i]) {
      const double n = ratio[i]*double(cells_per_period);
      if (!(n <= double(std::numeric_limits<int>::max())))
        throw Error("the grid would need more than 2147483647 cells along "+std::string(axis_names[i]));
      nCells[i] = static_cast<unsigned int>(n);
      continue;
    }
    const double n = std::ceil((hi[i]-lo[i])/side);
    if (!(n <= double(std::numeric_limits<int>::max())))
      throw Error("the grid would need more than 2147483647 cells along "+std::string(axis_names[i]));
    nCells[i] = (n >= 1.) ? static_cast<unsigned int>(n) : 1;
  }

  double grownLo[3], grownHi[3];
  for (int i=0; i<3; ++i) {
    if (opt.periodic[i]) {
      grownLo[i] = lo[i];
      grownHi[i] = hi[i];
      continue;
    }

    const double range = hi[i]-lo[i];
    double grid_span = nCells[i]*side;
    double offset = (grid_span-range)/2.;
    grownLo[i] = lo[i]-offset;
    grownHi[i] = hi[i]+offset;

    if (!(grownLo[i] <= lo[i] && grownHi[i] >= hi[i])) {
      if (nCells[i] >= static_cast<unsigned int>(std::numeric_limits<int>::max()))
        throw Error("the grid would need more than 2147483647 cells along "+std::string(axis_names[i]));
      nCells[i]++;
      grid_span = nCells[i]*side;
      offset = (grid_span-range)/2.;
      grownLo[i] = lo[i]-offset;
      grownHi[i] = hi[i]+offset;
    }

    if (!std::isfinite(grownLo[i]) || !std::isfinite(grownHi[i]))
      throw Error("the box is too large to represent");
  }

  const double extent[3] = {grownHi[0]-grownLo[0], grownHi[1]-grownLo[1], grownHi[2]-grownLo[2]};
  const double diagonal_sq = extent[0]*extent[0]+extent[1]*extent[1]+extent[2]*extent[2];
  if (!std::isfinite(diagonal_sq))
    throw Error("the box is too large: its squared diagonal overflows a double");

  const std::uint64_t nx = nCells[0], ny = nCells[1], nz = nCells[2];
  const char* const too_many_cells = "the grid would need more cells than 64-bit sizes can count or address";

  std::uint64_t total = 0, entries = 0;
  if (!checkedProduct(nx, ny, total) || !checkedProduct(total, nz, total) || !checkedProduct(total, 8, entries))
    throw Error(too_many_cells);
  if (total >= std::vector<size_t>().max_size() || total > std::vector<CellBlock>().max_size()
      || entries > std::vector<GridOffset>().max_size())
    throw Error(too_many_cells);

  double omega[3] = {0., 0., 0.};
  double slack_axis[3] = {0., 0., 0.};
  double slack = 0.;
  if (any_periodic) {
    double sum = 0.;
    for (int i=0; i<3; ++i) {
      const double span = double(nCells[i])*side;
      double err;
      if (opt.periodic[i]) {
        omega[i] = (std::fabs(period[i]-span)+0x1p-52*span)*pad_factor;
        err = periodic_slack*span+2.*omega[i];
        slack_axis[i] = err;
      }
      else err = plain_slack*span;
      const double cells = err/side;
      sum += cells*cells;
    }
    slack = std::sqrt(sum)*pad_factor;
  }

  m_cellsize = side;
  m_nCells = nCells;
  m_lims = {{{{grownLo[0], grownHi[0]}}, {{grownLo[1], grownHi[1]}}, {{grownLo[2], grownHi[2]}}}};
  m_periodic = opt.periodic;
  m_periodic_any = any_periodic;
  for (int i=0; i<3; ++i) {
    m_period[i] = period[i];
    m_omega[i] = omega[i];
    m_slack_axis[i] = slack_axis[i];
  }
  m_slack = slack;

  const int nC[3] = {static_cast<int>(m_nCells[0]), static_cast<int>(m_nCells[1]), static_cast<int>(m_nCells[2])};
  int first[3], last[3];
  std::uint64_t reach[3];
  for (int i=0; i<3; ++i) {
    first[i] = m_periodic[i] ? -(nC[i]/2) : -nC[i];
    last[i] = m_periodic[i] ? static_cast<int>((std::int64_t(nC[i])+1)/2) : nC[i];
    reach[i] = m_periodic[i] ? std::uint64_t(nC[i]/2) : std::uint64_t(nC[i]);
  }

  const std::uint64_t sum_sq = reach[0]*reach[0]+reach[1]*reach[1]+reach[2]*reach[2];
  const unsigned int dim_mask = static_cast<unsigned int>
    (std::ceil(std::sqrt(3.)+std::sqrt(double(sum_sq))));

  auto mask = std::make_shared<std::vector<std::vector<GridOffset>>>(dim_mask);

  for (int i=first[0]; i<last[0]; ++i) {
    const int abs_i = std::abs(i);
    const std::uint64_t dx = (abs_i > 0) ? abs_i-1 : 0;
    const std::uint64_t dx2 = dx*dx;
    for (int j=first[1]; j<last[1]; ++j) {
      const int abs_j = std::abs(j);
      const std::uint64_t dy = (abs_j > 0) ? abs_j-1 : 0;
      const std::uint64_t dy2 = dy*dy;
      for (int k=first[2]; k<last[2]; ++k) {
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
  m_cells.assign(total_cells, CellBlock{0, 0, 0});

  const size_t n = X.size();

  std::vector<size_t> cell(n);

  for (size_t i=0; i<n; ++i) {
    const auto coords = locate(m_lims, m_nCells, m_cellsize, X[i], Y[i], Z[i]);
    cell[i] = getLinearIndex(coords[0], coords[1], coords[2]);
    m_cells[cell[i]].cap++;
  }

  size_t start = 0;
  for (auto& block : m_cells) {
    block.start = start;
    start += block.cap;
  }

  m_X.resize(n);
  m_Y.resize(n);
  m_Z.resize(n);
  m_toPublic.resize(n);
  m_toInternal.resize(n);

  for (size_t i=0; i<n; ++i) {
    CellBlock& block = m_cells[cell[i]];
    const size_t slot = block.start+block.count++;
    m_X[slot] = X[i];
    m_Y[slot] = Y[i];
    m_Z[slot] = Z[i];
    m_toPublic[slot] = static_cast<unsigned int>(i);
    m_toInternal[i] = slot;
  }

  m_alive.assign(n, 1);
  m_nObjects = static_cast<unsigned int>(n);
}


// ===========================================================================


meshsearch::MeshGrid::MeshGrid (const std::vector<double>& X,
                                const std::vector<double>& Y,
                                const std::vector<double>& Z,
                                const double cellsize)
  : MeshGrid(X, Y, Z, cellsize, Options())
{}


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


bool meshsearch::MeshGrid::contains (const double X, const double Y, const double Z) const
{
  return inBox(m_nCells, m_lims, X, Y, Z);
}


// ===========================================================================


std::vector<std::uint8_t> meshsearch::MeshGrid::contains (const std::vector<double>& X,
                                                          const std::vector<double>& Y,
                                                          const std::vector<double>& Z) const
{
  std::vector<std::uint8_t> out;
  contains(X, Y, Z, out);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::contains (const std::vector<double>& X,
                                     const std::vector<double>& Y,
                                     const std::vector<double>& Z,
                                     std::vector<std::uint8_t>& out) const
{
  if (X.size() != Y.size() || X.size() != Z.size())
    throw Error("the coordinate vectors differ in length");
  out.clear();
  out.reserve(X.size());
  for (size_t i=0; i<X.size(); ++i) out.push_back(inBox(m_nCells, m_lims, X[i], Y[i], Z[i]) ? 1 : 0);
}


// ===========================================================================


std::array<std::array<double,2>,3> meshsearch::MeshGrid::get_lims () const
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


std::array<bool, 3> meshsearch::MeshGrid::get_periodic () const
{
  return m_periodic;
}


// ===========================================================================


template <class Threshold, class Visit>
void meshsearch::MeshGrid::walkPeriodic (const std::array<unsigned int, 3>& target,
                                         const double X, const double Y, const double Z, const double Rmin,
                                         Threshold&& threshold, Visit&& visit) const
{
  const CellFaces faces = periodicFaces(m_lims, m_nCells, m_cellsize, m_periodic, m_omega);

  std::int64_t cells[3], where[3];
  double period[3], half[3];
  for (int a=0; a<3; ++a) {
    cells[a] = m_nCells[a];
    where[a] = target[a];
    period[a] = m_periodic[a] ? m_period[a] : 0.;
    half[a] = m_periodic[a] ? m_period[a]*0.5 : std::numeric_limits<double>::infinity();
  }

  const unsigned int nShells = static_cast<unsigned int>(m_mask->size());

  unsigned int firstShellIndex = 0;
  const double Rmin_sq = Rmin*Rmin;
  if (Rmin_sq >= tiny_sq) {
    CellFaces start = faces;
    for (int a=0; a<3; ++a)
      if (m_periodic[a]) start.pad[a] = 2.*m_slack_axis[a];
    firstShellIndex = firstShell(start, nShells, Rmin, Rmin_sq);
  }

  for (unsigned int maskInd=firstShellIndex; maskInd<nShells; ++maskInd) {
    if (shellBeyondPeriodic(maskInd, m_cellsize, m_slack, threshold())) break;

    for (const auto& offset : (*m_mask)[maskInd]) {
      const std::int64_t step[3] = {offset.i, offset.j, offset.k};
      std::int64_t unwrapped[3], cell[3];
      double shift[3];
      bool inside = true, end = false, wrapped = false;

      for (int a=0; a<3; ++a) {
        const std::int64_t u = where[a]+step[a];
        unwrapped[a] = u;
        shift[a] = 0.;
        if (!m_periodic[a]) {
          if (u < 0 || u >= cells[a]) { inside = false; break; }
          cell[a] = u;
          continue;
        }
        if (u < 0) { cell[a] = u+cells[a]; shift[a] = period[a]; wrapped = true; }
        else if (u >= cells[a]) { cell[a] = u-cells[a]; shift[a] = -period[a]; wrapped = true; }
        else cell[a] = u;
        if (2*step[a] >= cells[a]-2 || -2*step[a] >= cells[a]-2) end = true;
      }
      if (!inside) continue;

      const CellBlock& block = m_cells[getLinearIndex(static_cast<unsigned int>(cell[0]), static_cast<unsigned int>(cell[1]), static_cast<unsigned int>(cell[2]))];
      if (block.count == 0) continue;
      const size_t first = block.start;
      const double* x = m_X.data()+first;
      const double* y = m_Y.data()+first;
      const double* z = m_Z.data()+first;
      auto visitSlot = [&] (const size_t k, const double dist_sq) { visit(first, k, dist_sq); };

      if (end) {
        scanImages(x, y, z, block.count, X, Y, Z, period, half, visitSlot);
        continue;
      }
      if (cellBeyond(faces, unwrapped[0], unwrapped[1], unwrapped[2], X, Y, Z, threshold())) continue;
      if (wrapped) scanShifted(x, y, z, block.count, X, Y, Z, shift[0], shift[1], shift[2], visitSlot);
      else scanBlock(x, y, z, block.count, X, Y, Z, visitSlot);
    }
  }
}


// ===========================================================================


unsigned int meshsearch::MeshGrid::nearestObjectPeriodic (double X, double Y, double Z, const unsigned int* index) const
{
  size_t self = no_slot;
  if (index) {
    checkIndex(m_alive, *index);
    if (m_nObjects < 2) throw Error("no other object remains in the grid");
    self = m_toInternal[*index];
    X = m_X[self];
    Y = m_Y[self];
    Z = m_Z[self];
  }
  const auto target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  if (!index && m_nObjects == 0) throw Error("the grid holds no objects");

  double min_dist_sq = std::numeric_limits<double>::max();
  size_t finalSlot = no_slot;

  walkPeriodic(target, X, Y, Z, 0., [&] { return min_dist_sq; },
               [&] (const size_t first, const size_t k, const double dist_sq) {
                 if (first+k == self) return;
                 if (finalSlot == no_slot || dist_sq < min_dist_sq) {
                   min_dist_sq = dist_sq;
                   finalSlot = first+k;
                 }
               });

  if (finalSlot == no_slot)
    throw Error("no object found");

  return m_toPublic[finalSlot];
}


// ===========================================================================


void meshsearch::MeshGrid::nearestObjectsPeriodic (std::vector<unsigned int>& out, const unsigned int N,
                                                    double X, double Y, double Z,
                                                    const unsigned int* index) const
{
  size_t self = no_slot;
  if (index) {
    checkIndex(m_alive, *index);
    if (N >= m_nObjects) throw Error("N is larger than the number of available neighbours");
    if (N == 0) {
      out.clear();
      return;
    }
    self = m_toInternal[*index];
    X = m_X[self];
    Y = m_Y[self];
    Z = m_Z[self];
  }
  const auto target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  if (!index) {
    if (N > m_nObjects) throw Error("N is larger than the number of objects");
  }
  out.clear();
  if (N == 0) return;

  auto walk = [&] (auto& pq)
  {
    walkPeriodic(target, X, Y, Z, 0., [&] { return pq.T; },
                 [&] (const size_t first, const size_t k, const double dist_sq) {
                   if (first+k != self && dist_sq <= pq.T) pq.offer(dist_sq, m_toPublic[first+k]);
                 });
  };

  if (N <= sorted_limit) {
    SortedKeep pq(N);
    walk(pq);
    pq.finish(out);
  }
  else {
    BufferKeep pq(N);
    walk(pq);
    pq.finish(out);
  }
}


// ===========================================================================


void meshsearch::MeshGrid::closeObjectsPeriodic (std::vector<unsigned int>& out,
                                                  double X, double Y, double Z,
                                                  const double Rmax, const double Rmin,
                                                  const unsigned int* index) const
{
  size_t self = no_slot;
  std::array<unsigned int, 3> target;
  if (index) {
    checkIndex(m_alive, *index);
    checkRadii(Rmax, Rmin);
    self = m_toInternal[*index];
    X = m_X[self];
    Y = m_Y[self];
    Z = m_Z[self];
    target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  }
  else {
    target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
    checkRadii(Rmax, Rmin);
  }

  const double Rmax_sq = Rmax*Rmax;
  const double Rmin_sq = Rmin*Rmin;

  std::vector<unsigned int> found;
  found.swap(out);
  found.clear();

  walkPeriodic(target, X, Y, Z, Rmin, [&] { return Rmax_sq; },
               [&] (const size_t first, const size_t k, const double dist_sq) {
                 if (first+k != self && dist_sq <= Rmax_sq && dist_sq >= Rmin_sq) found.push_back(m_toPublic[first+k]);
               });

  out.swap(found);
}


// ===========================================================================


template <class Threshold, class Visit>
void meshsearch::MeshGrid::walk (const std::array<unsigned int, 3>& target,
                                 const double X, const double Y, const double Z, const double Rmin,
                                 Threshold&& threshold, Visit&& visit) const
{
  if (m_periodic_any) walkPeriodic(target, X, Y, Z, Rmin, threshold, visit);
  else walkPlain(target, X, Y, Z, Rmin, threshold, visit);
}


// ===========================================================================


template <class Threshold, class Visit>
void meshsearch::MeshGrid::walkPlain (const std::array<unsigned int, 3>& target,
                                      const double X, const double Y, const double Z, const double Rmin,
                                      Threshold&& threshold, Visit&& visit) const
{
  const int targetX = static_cast<int>(target[0]);
  const int targetY = static_cast<int>(target[1]);
  const int targetZ = static_cast<int>(target[2]);

  const int nCx = static_cast<int>(m_nCells[0]);
  const int nCy = static_cast<int>(m_nCells[1]);
  const int nCz = static_cast<int>(m_nCells[2]);

  const CellFaces faces = cellFaces(m_lims, m_nCells, m_cellsize);

  const unsigned int nShells = static_cast<unsigned int>(m_mask->size());
  const unsigned int min_mask_idx = firstShell(faces, nShells, Rmin, Rmin*Rmin);

  for (unsigned int maskInd=min_mask_idx; maskInd<nShells; maskInd++) {
    if (shellBeyond(maskInd, m_cellsize, threshold())) break;

    for (const auto& offset : (*m_mask)[maskInd]) {
      const std::int64_t currX = std::int64_t(targetX)+offset.i;
      const std::int64_t currY = std::int64_t(targetY)+offset.j;
      const std::int64_t currZ = std::int64_t(targetZ)+offset.k;
      if (currX >= 0 && currX < nCx && currY >= 0 && currY < nCy && currZ >= 0 && currZ < nCz) {
        const CellBlock& block = m_cells[getLinearIndex(static_cast<unsigned int>(currX), static_cast<unsigned int>(currY), static_cast<unsigned int>(currZ))];
        if (block.count == 0 || cellBeyond(faces, currX, currY, currZ, X, Y, Z, threshold())) continue;
        const size_t first = block.start;
        scanBlock(m_X.data()+first, m_Y.data()+first, m_Z.data()+first, block.count, X, Y, Z,
                  [&] (const size_t k, const double dist_sq) { visit(first, k, dist_sq); });
      }
    }
  }
}


// ===========================================================================


template <bool Exclude, class Sink>
void meshsearch::MeshGrid::closeScan (const std::array<unsigned int, 3>& target,
                                      const double X, const double Y, const double Z, const size_t self,
                                      const double Rmax, const double Rmin, Sink&& sink) const
{
  const double Rmax_sq = Rmax*Rmax;
  const double Rmin_sq = Rmin*Rmin;

  walk(target, X, Y, Z, Rmin, [&] { return Rmax_sq; },
       [&] (const size_t first, const size_t k, const double dist_sq) {
         if ((!Exclude || first+k != self) && dist_sq <= Rmax_sq && dist_sq >= Rmin_sq) sink(m_toPublic[first+k]);
       });
}


// ===========================================================================


unsigned int meshsearch::MeshGrid::nearestObject (const double X, const double Y, const double Z) const
{
  if (m_periodic_any) return nearestObjectPeriodic(X, Y, Z, nullptr);

  const auto target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  if (m_nObjects == 0) throw Error("the grid holds no objects");

  const int targetX = static_cast<int>(target[0]);
  const int targetY = static_cast<int>(target[1]);
  const int targetZ = static_cast<int>(target[2]);

  const int nCx = static_cast<int>(m_nCells[0]);
  const int nCy = static_cast<int>(m_nCells[1]);
  const int nCz = static_cast<int>(m_nCells[2]);

  const CellFaces faces = cellFaces(m_lims, m_nCells, m_cellsize);

  double min_dist_sq = std::numeric_limits<double>::max();
  size_t finalSlot = no_slot;
  unsigned int maskInd = 0;

  while (maskInd < m_mask->size()) {
    if (shellBeyond(maskInd, m_cellsize, min_dist_sq)) break;

    for (const auto& offset : (*m_mask)[maskInd]) {
      const std::int64_t currX = std::int64_t(targetX)+offset.i;
      const std::int64_t currY = std::int64_t(targetY)+offset.j;
      const std::int64_t currZ = std::int64_t(targetZ)+offset.k;
      if (currX >= 0 && currX < nCx && currY >= 0 && currY < nCy && currZ >= 0 && currZ < nCz) {
        const CellBlock& block = m_cells[getLinearIndex(static_cast<unsigned int>(currX), static_cast<unsigned int>(currY), static_cast<unsigned int>(currZ))];
        if (block.count == 0 || cellBeyond(faces, currX, currY, currZ, X, Y, Z, min_dist_sq)) continue;
        const size_t first = block.start;
        scanBlock(m_X.data()+first, m_Y.data()+first, m_Z.data()+first, block.count, X, Y, Z,
                  [&] (const size_t k, const double dist_sq) {
                    if (finalSlot == no_slot || dist_sq < min_dist_sq) {
                      min_dist_sq = dist_sq;
                      finalSlot = first+k;
                    }
                  });
      }
    }
    maskInd++;
  }

  if (finalSlot == no_slot)
    throw Error("no object found");

  return m_toPublic[finalSlot];
}


// ===========================================================================


unsigned int meshsearch::MeshGrid::nearestObject (const unsigned int index) const
{
  if (m_periodic_any) return nearestObjectPeriodic(0., 0., 0., &index);

  checkIndex(m_alive, index);
  if (m_nObjects < 2) throw Error("no other object remains in the grid");

  const size_t self = m_toInternal[index];
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

  const CellFaces faces = cellFaces(m_lims, m_nCells, m_cellsize);

  double min_dist_sq = std::numeric_limits<double>::max();
  size_t finalSlot = no_slot;
  unsigned int maskInd = 0;

  while (maskInd < m_mask->size()) {
    if (shellBeyond(maskInd, m_cellsize, min_dist_sq)) break;

    for (const auto& offset : (*m_mask)[maskInd]) {
      const std::int64_t currX = std::int64_t(targetX)+offset.i;
      const std::int64_t currY = std::int64_t(targetY)+offset.j;
      const std::int64_t currZ = std::int64_t(targetZ)+offset.k;
      if (currX >= 0 && currX < nCx && currY >= 0 && currY < nCy && currZ >= 0 && currZ < nCz) {
        const CellBlock& block = m_cells[getLinearIndex(static_cast<unsigned int>(currX), static_cast<unsigned int>(currY), static_cast<unsigned int>(currZ))];
        if (block.count == 0 || cellBeyond(faces, currX, currY, currZ, X, Y, Z, min_dist_sq)) continue;
        const size_t first = block.start;
        scanBlock(m_X.data()+first, m_Y.data()+first, m_Z.data()+first, block.count, X, Y, Z,
                  [&] (const size_t k, const double dist_sq) {
                    if (first+k == self) return;
                    if (finalSlot == no_slot || dist_sq < min_dist_sq) {
                      min_dist_sq = dist_sq;
                      finalSlot = first+k;
                    }
                  });
      }
    }
    maskInd++;
  }

  if (finalSlot == no_slot)
    throw Error("no object found");

  return m_toPublic[finalSlot];
}


// ===========================================================================


std::vector<unsigned int> meshsearch::MeshGrid::nearestObjects (const unsigned int N,
                                                                const double X, const double Y, const double Z) const
{
  std::vector<unsigned int> out;
  nearestObjects(N, X, Y, Z, out);
  return out;
}


// ===========================================================================


std::vector<unsigned int> meshsearch::MeshGrid::nearestObjects (const unsigned int N,
                                                                const unsigned int index) const
{
  std::vector<unsigned int> out;
  nearestObjects(N, index, out);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::nearestObjects (const unsigned int N, const double X, const double Y, const double Z,
                                           std::vector<unsigned int>& out) const
{
  if (m_periodic_any) return nearestObjectsPeriodic(out, N, X, Y, Z, nullptr);

  const auto target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  if (N > m_nObjects) throw Error("N is larger than the number of objects");
  out.clear();
  if (N == 0) return;

  const int targetX = static_cast<int>(target[0]);
  const int targetY = static_cast<int>(target[1]);
  const int targetZ = static_cast<int>(target[2]);

  const int nCx = static_cast<int>(m_nCells[0]);
  const int nCy = static_cast<int>(m_nCells[1]);
  const int nCz = static_cast<int>(m_nCells[2]);

  const CellFaces faces = cellFaces(m_lims, m_nCells, m_cellsize);

  auto walk = [&] (auto& pq)
  {
    unsigned int maskInd = 0;

    while (maskInd < m_mask->size()) {
      if (shellBeyond(maskInd, m_cellsize, pq.T)) break;

      for (const auto& offset : (*m_mask)[maskInd]) {
        const std::int64_t currX = std::int64_t(targetX)+offset.i;
        const std::int64_t currY = std::int64_t(targetY)+offset.j;
        const std::int64_t currZ = std::int64_t(targetZ)+offset.k;
        if (currX >= 0 && currX < nCx && currY >= 0 && currY < nCy && currZ >= 0 && currZ < nCz) {
          const CellBlock& block = m_cells[getLinearIndex(static_cast<unsigned int>(currX), static_cast<unsigned int>(currY), static_cast<unsigned int>(currZ))];
          if (block.count == 0 || cellBeyond(faces, currX, currY, currZ, X, Y, Z, pq.T)) continue;
          const size_t first = block.start;
          const unsigned int* pub = m_toPublic.data()+first;
          scanBlock(m_X.data()+first, m_Y.data()+first, m_Z.data()+first, block.count, X, Y, Z,
                    [&] (const size_t k, const double dist_sq) {
                      if (dist_sq <= pq.T) pq.offer(dist_sq, pub[k]);
                    });
        }
      }
      maskInd++;
    }
  };

  if (N <= sorted_limit) {
    SortedKeep pq(N);
    walk(pq);
    pq.finish(out);
  }
  else {
    BufferKeep pq(N);
    walk(pq);
    pq.finish(out);
  }
}


// ===========================================================================


void meshsearch::MeshGrid::nearestObjects (const unsigned int N, const unsigned int index,
                                           std::vector<unsigned int>& out) const
{
  if (m_periodic_any) return nearestObjectsPeriodic(out, N, 0., 0., 0., &index);

  checkIndex(m_alive, index);
  if (N >= m_nObjects) throw Error("N is larger than the number of available neighbours");
  out.clear();
  if (N == 0) return;

  const size_t self = m_toInternal[index];
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

  const CellFaces faces = cellFaces(m_lims, m_nCells, m_cellsize);

  auto walk = [&] (auto& pq)
  {
    unsigned int maskInd = 0;

    while (maskInd < m_mask->size()) {
      if (shellBeyond(maskInd, m_cellsize, pq.T)) break;

      for (const auto& offset : (*m_mask)[maskInd]) {
        const std::int64_t currX = std::int64_t(targetX)+offset.i;
        const std::int64_t currY = std::int64_t(targetY)+offset.j;
        const std::int64_t currZ = std::int64_t(targetZ)+offset.k;
        if (currX >= 0 && currX < nCx && currY >= 0 && currY < nCy && currZ >= 0 && currZ < nCz) {
          const CellBlock& block = m_cells[getLinearIndex(static_cast<unsigned int>(currX), static_cast<unsigned int>(currY), static_cast<unsigned int>(currZ))];
          if (block.count == 0 || cellBeyond(faces, currX, currY, currZ, X, Y, Z, pq.T)) continue;
          const size_t first = block.start;
          const unsigned int* pub = m_toPublic.data()+first;
          scanBlock(m_X.data()+first, m_Y.data()+first, m_Z.data()+first, block.count, X, Y, Z,
                    [&] (const size_t k, const double dist_sq) {
                      if (first+k != self && dist_sq <= pq.T) pq.offer(dist_sq, pub[k]);
                    });
        }
      }
      maskInd++;
    }
  };

  if (N <= sorted_limit) {
    SortedKeep pq(N);
    walk(pq);
    pq.finish(out);
  }
  else {
    BufferKeep pq(N);
    walk(pq);
    pq.finish(out);
  }
}


// ===========================================================================


std::vector<unsigned int> meshsearch::MeshGrid::closeObjects (const double X, const double Y, const double Z,
                                                              const double Rmax, const double Rmin) const
{
  std::vector<unsigned int> out;
  closeObjects(X, Y, Z, Rmax, Rmin, out);
  return out;
}


// ===========================================================================


std::vector<unsigned int> meshsearch::MeshGrid::closeObjects (const unsigned int index,
                                                              const double Rmax, const double Rmin) const
{
  std::vector<unsigned int> out;
  closeObjects(index, Rmax, Rmin, out);
  return out;
}


// ===========================================================================


void meshsearch::MeshGrid::closeObjects (const double X, const double Y, const double Z,
                                         const double Rmax, const double Rmin,
                                         std::vector<unsigned int>& out) const
{
  if (m_periodic_any) return closeObjectsPeriodic(out, X, Y, Z, Rmax, Rmin, nullptr);

  const auto target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  checkRadii(Rmax, Rmin);

  const double Rmax_sq = Rmax*Rmax;
  const double Rmin_sq = Rmin*Rmin;

  const int targetX = static_cast<int>(target[0]);
  const int targetY = static_cast<int>(target[1]);
  const int targetZ = static_cast<int>(target[2]);

  const int nCx = static_cast<int>(m_nCells[0]);
  const int nCy = static_cast<int>(m_nCells[1]);
  const int nCz = static_cast<int>(m_nCells[2]);

  const CellFaces faces = cellFaces(m_lims, m_nCells, m_cellsize);

  const unsigned int nShells = static_cast<unsigned int>(m_mask->size());
  const unsigned int min_mask_idx = firstShell(faces, nShells, Rmin, Rmin_sq);

  std::vector<unsigned int> found;
  found.swap(out);
  found.clear();

  for (unsigned int maskInd=min_mask_idx; maskInd<nShells; maskInd++) {
    if (shellBeyond(maskInd, m_cellsize, Rmax_sq)) break;

    for (const auto& offset : (*m_mask)[maskInd]) {
      const std::int64_t currX = std::int64_t(targetX)+offset.i;
      const std::int64_t currY = std::int64_t(targetY)+offset.j;
      const std::int64_t currZ = std::int64_t(targetZ)+offset.k;
      if (currX >= 0 && currX < nCx && currY >= 0 && currY < nCy && currZ >= 0 && currZ < nCz) {
        const CellBlock& block = m_cells[getLinearIndex(static_cast<unsigned int>(currX), static_cast<unsigned int>(currY), static_cast<unsigned int>(currZ))];
        if (block.count == 0 || cellBeyond(faces, currX, currY, currZ, X, Y, Z, Rmax_sq)) continue;
        const size_t first = block.start;
        const unsigned int* pub = m_toPublic.data()+first;
        scanBlock(m_X.data()+first, m_Y.data()+first, m_Z.data()+first, block.count, X, Y, Z,
                  [&] (const size_t k, const double dist_sq) {
                    if (dist_sq <= Rmax_sq && dist_sq >= Rmin_sq) found.push_back(pub[k]);
                  });
      }
    }
  }

  out.swap(found);
}


// ===========================================================================


void meshsearch::MeshGrid::closeObjects (const double X, const double Y, const double Z, const double Rmax,
                                         std::vector<unsigned int>& out) const
{
  closeObjects(X, Y, Z, Rmax, 0., out);
}


// ===========================================================================


void meshsearch::MeshGrid::closeObjects (const unsigned int index, const double Rmax, const double Rmin,
                                         std::vector<unsigned int>& out) const
{
  if (m_periodic_any) return closeObjectsPeriodic(out, 0., 0., 0., Rmax, Rmin, &index);

  checkIndex(m_alive, index);
  checkRadii(Rmax, Rmin);

  const size_t self = m_toInternal[index];
  const double X = m_X[self];
  const double Y = m_Y[self];
  const double Z = m_Z[self];

  const double Rmax_sq = Rmax*Rmax;
  const double Rmin_sq = Rmin*Rmin;

  const auto target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  const int targetX = static_cast<int>(target[0]);
  const int targetY = static_cast<int>(target[1]);
  const int targetZ = static_cast<int>(target[2]);

  const int nCx = static_cast<int>(m_nCells[0]);
  const int nCy = static_cast<int>(m_nCells[1]);
  const int nCz = static_cast<int>(m_nCells[2]);

  const CellFaces faces = cellFaces(m_lims, m_nCells, m_cellsize);

  const unsigned int nShells = static_cast<unsigned int>(m_mask->size());
  const unsigned int min_mask_idx = firstShell(faces, nShells, Rmin, Rmin_sq);

  std::vector<unsigned int> found;
  found.swap(out);
  found.clear();

  for (unsigned int maskInd=min_mask_idx; maskInd<nShells; maskInd++) {
    if (shellBeyond(maskInd, m_cellsize, Rmax_sq)) break;

    for (const auto& offset : (*m_mask)[maskInd]) {
      const std::int64_t currX = std::int64_t(targetX)+offset.i;
      const std::int64_t currY = std::int64_t(targetY)+offset.j;
      const std::int64_t currZ = std::int64_t(targetZ)+offset.k;
      if (currX >= 0 && currX < nCx && currY >= 0 && currY < nCy && currZ >= 0 && currZ < nCz) {
        const CellBlock& block = m_cells[getLinearIndex(static_cast<unsigned int>(currX), static_cast<unsigned int>(currY), static_cast<unsigned int>(currZ))];
        if (block.count == 0 || cellBeyond(faces, currX, currY, currZ, X, Y, Z, Rmax_sq)) continue;
        const size_t first = block.start;
        const unsigned int* pub = m_toPublic.data()+first;
        scanBlock(m_X.data()+first, m_Y.data()+first, m_Z.data()+first, block.count, X, Y, Z,
                  [&] (const size_t k, const double dist_sq) {
                    if (first+k != self && dist_sq <= Rmax_sq && dist_sq >= Rmin_sq) found.push_back(pub[k]);
                  });
      }
    }
  }

  out.swap(found);
}


// ===========================================================================


void meshsearch::MeshGrid::closeObjects (const unsigned int index, const double Rmax,
                                         std::vector<unsigned int>& out) const
{
  closeObjects(index, Rmax, 0., out);
}


// ===========================================================================


std::size_t meshsearch::MeshGrid::countObjects (const double X, const double Y, const double Z,
                                                const double Rmax, const double Rmin) const
{
  const auto target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  checkRadii(Rmax, Rmin);

  std::size_t count = 0;
  closeScan<false>(target, X, Y, Z, no_slot, Rmax, Rmin, [&] (const unsigned int) { ++count; });
  return count;
}


// ===========================================================================


std::size_t meshsearch::MeshGrid::countObjects (const unsigned int index,
                                                const double Rmax, const double Rmin) const
{
  checkIndex(m_alive, index);
  checkRadii(Rmax, Rmin);

  const size_t self = m_toInternal[index];
  const double X = m_X[self];
  const double Y = m_Y[self];
  const double Z = m_Z[self];

  const auto target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);

  std::size_t count = 0;
  closeScan<true>(target, X, Y, Z, self, Rmax, Rmin, [&] (const unsigned int) { ++count; });
  return count;
}


// ===========================================================================


size_t meshsearch::MeshGrid::get_cell (const double X, const double Y, const double Z) const
{
  const auto coords = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  return getLinearIndex(coords[0], coords[1], coords[2]);
}


// ===========================================================================


std::array<unsigned int, 3> meshsearch::MeshGrid::get_cellCoords (const double X, const double Y, const double Z) const
{
  return locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
}


// ===========================================================================


std::vector<unsigned int> meshsearch::MeshGrid::get_objectsInCell (const double X, const double Y, const double Z) const
{
  const auto coords = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  const CellBlock& block = m_cells[getLinearIndex(coords[0], coords[1], coords[2])];
  const auto first = m_toPublic.begin()+static_cast<std::ptrdiff_t>(block.start);
  return std::vector<unsigned int>(first, first+block.count);
}


// ===========================================================================


void meshsearch::MeshGrid::removeObject (const unsigned int index)
{
  checkIndex(m_alive, index);

  const size_t slot = m_toInternal[index];

  const auto coords = locate(m_lims, m_nCells, m_cellsize, m_X[slot], m_Y[slot], m_Z[slot]);
  CellBlock& block = m_cells[getLinearIndex(coords[0], coords[1], coords[2])];

  const size_t end = block.start+block.count;
  for (size_t k=slot+1; k<end; ++k) {
    m_X[k-1] = m_X[k];
    m_Y[k-1] = m_Y[k];
    m_Z[k-1] = m_Z[k];
    m_toPublic[k-1] = m_toPublic[k];
    m_toInternal[m_toPublic[k-1]] = k-1;
  }
  block.count--;

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
  const size_t target = getLinearIndex(coords[0], coords[1], coords[2]);

  if (m_cells[target].count == m_cells[target].cap) {
    CellBlock& block = m_cells[target];
    const unsigned int count = block.count;
    const unsigned int cap = (count >= max_objects/2) ? max_objects : std::max(2*count, min_block);
    const size_t slots = m_X.size();
    const size_t base = (block.start+block.cap == slots) ? block.start : slots;
    const size_t grown = base+cap;

    try {
      m_X.resize(grown);
      m_Y.resize(grown);
      m_Z.resize(grown);
      m_toPublic.resize(grown);
    }
    catch (...) {
      m_X.resize(slots);
      m_Y.resize(slots);
      m_Z.resize(slots);
      m_toPublic.resize(slots);
      throw;
    }

    if (base != block.start) {
      for (size_t k=0; k<count; ++k) {
        m_X[base+k] = m_X[block.start+k];
        m_Y[base+k] = m_Y[block.start+k];
        m_Z[base+k] = m_Z[block.start+k];
        m_toPublic[base+k] = m_toPublic[block.start+k];
        m_toInternal[m_toPublic[base+k]] = base+k;
      }
      block.start = base;
    }
    block.cap = cap;
  }

  CellBlock& block = m_cells[target];
  const size_t slot = block.start+block.count;

  m_toInternal.push_back(slot);
  try {
    m_alive.push_back(1);
  }
  catch (...) {
    m_toInternal.pop_back();
    throw;
  }

  m_X[slot] = X;
  m_Y[slot] = Y;
  m_Z[slot] = Z;
  m_toPublic[slot] = index;
  block.count++;
  m_nObjects++;

  return index;
}
