#ifndef MESHSEARCH_MESHGRIDWALK_H
#define MESHSEARCH_MESHGRIDWALK_H

#include "meshsearch/MeshGrid.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

  const double tiny_sq = 0x1p-1000;

  const double huge_sq = 0x1p1000;

  const double stop_factor = 1.-0x1p-14;

  const double pad_scale = 0x1p-49;

  const double pad_factor = 1.+0x1p-40;

  const double skip_factor = 1.-0x1p-40;

  const double far_reach = 3.46413;

  const size_t scan_chunk = 64;

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

  using Lims = std::array<std::array<double,2>,3>;

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

  bool axisBeyond (const CellFaces& faces, const int axis, const std::int64_t cell, const double q, const double threshold)
  {
    const double gap = axisGap(faces, axis, cell, q);
    const double gap_sq = gap*gap;
    return gap_sq >= tiny_sq && gap_sq*skip_factor > threshold;
  }

  class StepWindow {
  public:
    std::int64_t lo[3], hi[3];

    StepWindow (const CellFaces& faces, const std::int64_t* target, const double X, const double Y, const double Z,
                const std::int64_t* first, const std::int64_t* last)
      : m_faces(faces)
    {
      m_q[0] = X;
      m_q[1] = Y;
      m_q[2] = Z;
      for (int a=0; a<3; ++a) {
        m_target[a] = target[a];
        m_first[a] = first[a];
        m_last[a] = last[a];
      }
    }

    void update (const double threshold, const bool reach)
    {
      if (m_set && threshold == m_threshold && reach == m_reach) return;
      m_set = true;
      m_threshold = threshold;
      m_reach = reach;
      const bool bounded = reach && threshold <= huge_sq;
      for (int a=0; a<3; ++a) {
        std::int64_t l = m_first[a], h = m_last[a];
        if (bounded) axisRange(a, threshold, l, h);
        lo[a] = l-m_target[a];
        hi[a] = h-m_target[a];
      }
    }

  private:
    void axisRange (const int a, const double threshold, std::int64_t& l, std::int64_t& h) const
    {
      const std::int64_t t = m_target[a];
      const double q = m_q[a];
      if (axisBeyond(m_faces, a, t, q, threshold)) return;
      const double reach = (std::sqrt(threshold)+m_faces.pad[a])/m_faces.cellsize;
      const double pos = (q-m_faces.lo[a])/m_faces.cellsize;
      const double up = std::floor(pos+reach);
      const double down = std::floor(pos-reach);
      if (up < double(h)) h = (up > double(t)) ? static_cast<std::int64_t>(up) : t;
      while (h < m_last[a] && !axisBeyond(m_faces, a, h+1, q, threshold)) ++h;
      while (h > t && axisBeyond(m_faces, a, h, q, threshold)) --h;
      if (down > double(l)) l = (down < double(t)) ? static_cast<std::int64_t>(down) : t;
      while (l > m_first[a] && !axisBeyond(m_faces, a, l-1, q, threshold)) --l;
      while (l < t && axisBeyond(m_faces, a, l, q, threshold)) ++l;
    }

    const CellFaces& m_faces;
    double m_q[3];
    std::int64_t m_target[3], m_first[3], m_last[3];
    double m_threshold = 0.;
    bool m_reach = false;
    bool m_set = false;
  };

  StepWindow plainWindow (const CellFaces& faces, const std::array<unsigned int, 3>& target,
                          const std::array<unsigned int, 3>& nCells, const double X, const double Y, const double Z)
  {
    const std::int64_t where[3] = {target[0], target[1], target[2]};
    const std::int64_t first[3] = {0, 0, 0};
    const std::int64_t last[3] = {std::int64_t(nCells[0])-1, std::int64_t(nCells[1])-1, std::int64_t(nCells[2])-1};
    return StepWindow(faces, where, X, Y, Z, first, last);
  }


  template <class Offset>
  class ShellSegments {
  public:
    const Offset* begin = nullptr;
    const Offset* end = nullptr;

    ShellSegments (const std::vector<Offset>& shell, const std::vector<size_t>& runs, const StepWindow& w,
                   const unsigned int shellIndex)
      : m_e(shell.data()), m_runs(runs.data()), m_nRuns(runs.size()-1),
        m_ihi(w.hi[0]), m_jlo(w.lo[1]), m_jhi(w.hi[1]), m_klo(w.lo[2]), m_khi(w.hi[2])
    {
      const std::int64_t extent = std::int64_t(shellIndex)+1;
      if (w.lo[0] <= -extent && w.hi[0] >= extent && w.lo[1] <= -extent && w.hi[1] >= extent) {
        m_whole = true;
        return;
      }
      size_t a = 0, b = m_nRuns;
      while (a < b) {
        const size_t m = (a+b)/2;
        if (m_e[m_runs[m]].i < w.lo[0]) a = m+1;
        else b = m;
      }
      m_run = a;
    }

    bool next ()
    {
      if (m_whole) {
        if (m_run > 0 || m_nRuns == 0) return false;
        m_run = 1;
        begin = m_e;
        end = m_e+m_runs[m_nRuns];
        return true;
      }
      for (; m_run<m_nRuns; ++m_run) {
        size_t p = m_runs[m_run];
        if (m_e[p].i > m_ihi) break;
        size_t q = m_runs[m_run+1];
        if (m_e[p].j < m_jlo) {
          size_t hi = q;
          while (p < hi) {
            const size_t m = (p+hi)/2;
            if (m_e[m].j < m_jlo) p = m+1;
            else hi = m;
          }
        }
        if (p < q && m_e[q-1].j > m_jhi) {
          size_t lo = p;
          while (lo < q) {
            const size_t m = (lo+q)/2;
            if (m_e[m].j <= m_jhi) lo = m+1;
            else q = m;
          }
        }
        if (p == q) continue;
        ++m_run;
        begin = m_e+p;
        end = m_e+q;
        return true;
      }
      return false;
    }

    bool holds (const Offset& entry) const
    {
      return entry.k >= m_klo && entry.k <= m_khi;
    }

  private:
    const Offset* m_e;
    const size_t* m_runs;
    size_t m_nRuns;
    std::int64_t m_ihi, m_jlo, m_jhi, m_klo, m_khi;
    bool m_whole = false;
    size_t m_run = 0;
  };

  template <class Visit, class = void>
  struct WholeRules : std::false_type {};

  template <class Visit>
  struct WholeRules<Visit, std::void_t<typename Visit::whole_rules>> : std::true_type {};

}


// ===========================================================================


template <class Threshold, class Visit>
void meshsearch::MeshGrid::walkPeriodic (const std::array<unsigned int, 3>& target,
                                         const double X, const double Y, const double Z, const double Rmin,
                                         Threshold&& threshold, Visit&& visit) const
{
  constexpr bool rules = WholeRules<std::decay_t<Visit>>::value;

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

  std::int64_t first[3], last[3], endLimit = std::numeric_limits<std::int64_t>::max();
  for (int a=0; a<3; ++a) {
    if (m_periodic[a]) {
      first[a] = where[a]-cells[a]/2;
      last[a] = where[a]+(cells[a]+1)/2-1;
      endLimit = std::min(endLimit, cells[a]-3);
    }
    else {
      first[a] = 0;
      last[a] = cells[a]-1;
    }
  }
  StepWindow window(faces, where, X, Y, Z, first, last);

  unsigned int wholeEnd = 0;
  if constexpr (rules) {
    CellFaces start = faces;
    for (int a=0; a<3; ++a)
      if (m_periodic[a]) start.pad[a] = 2.*m_slack_axis[a];
    wholeEnd = visit.shells(start, nShells);
  }

  for (unsigned int maskInd=firstShellIndex; maskInd<nShells; ++maskInd) {
    if (shellBeyondPeriodic(maskInd, m_cellsize, m_slack, threshold())) break;
    window.update(threshold(), 2*(std::int64_t(maskInd)+1) <= endLimit);
    bool wholeShell = false, verdicts = false;
    if constexpr (rules) {
      wholeShell = maskInd < wholeEnd && 2*(std::int64_t(maskInd)+1) <= endLimit
        && visit.shellClear(maskInd, m_cellsize, m_slack);
      verdicts = visit.shellVerdicts(maskInd, m_cellsize, m_slack);
    }

    ShellSegments<GridOffset> segments((*m_mask)[maskInd], (*m_runs)[maskInd], window, maskInd);
    while (segments.next()) {
      for (const GridOffset* entry=segments.begin; entry!=segments.end; ++entry) {
        if (!segments.holds(*entry)) continue;
        const std::int64_t step[3] = {entry->i, entry->j, entry->k};
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

        const CellBlock& block = m_cells[size_t(static_cast<unsigned int>(cell[0]))*size_t(m_nCells[1])*size_t(m_nCells[2])+size_t(static_cast<unsigned int>(cell[1]))*size_t(m_nCells[2])+size_t(static_cast<unsigned int>(cell[2]))];
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
        if constexpr (rules) {
          if (wholeShell) {
            visit.emit(first, block.count);
            continue;
          }
          if (verdicts ? visit.settle(faces, unwrapped[0], unwrapped[1], unwrapped[2], X, Y, Z, first, block.count)
                       : cellBeyond(faces, unwrapped[0], unwrapped[1], unwrapped[2], X, Y, Z, threshold())) continue;
        }
        else if (cellBeyond(faces, unwrapped[0], unwrapped[1], unwrapped[2], X, Y, Z, threshold())) continue;
        if (wrapped) scanShifted(x, y, z, block.count, X, Y, Z, shift[0], shift[1], shift[2], visitSlot);
        else scanBlock(x, y, z, block.count, X, Y, Z, visitSlot);
      }
    }
  }
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
  constexpr bool rules = WholeRules<std::decay_t<Visit>>::value;

  const int targetX = static_cast<int>(target[0]);
  const int targetY = static_cast<int>(target[1]);
  const int targetZ = static_cast<int>(target[2]);

  const CellFaces faces = cellFaces(m_lims, m_nCells, m_cellsize);
  StepWindow window = plainWindow(faces, target, m_nCells, X, Y, Z);

  const unsigned int nShells = static_cast<unsigned int>(m_mask->size());
  const unsigned int min_mask_idx = firstShell(faces, nShells, Rmin, Rmin*Rmin);

  unsigned int wholeEnd = 0;
  if constexpr (rules) wholeEnd = visit.shells(faces, nShells);

  for (unsigned int maskInd=min_mask_idx; maskInd<nShells; maskInd++) {
    if (shellBeyond(maskInd, m_cellsize, threshold())) break;
    window.update(threshold(), true);
    bool wholeShell = false, verdicts = false;
    if constexpr (rules) {
      wholeShell = maskInd < wholeEnd && visit.shellClear(maskInd, m_cellsize, 0.);
      verdicts = visit.shellVerdicts(maskInd, m_cellsize, 0.);
    }

    ShellSegments<GridOffset> segments((*m_mask)[maskInd], (*m_runs)[maskInd], window, maskInd);
    while (segments.next()) {
      for (const GridOffset* entry=segments.begin; entry!=segments.end; ++entry) {
        if (!segments.holds(*entry)) continue;
        const std::int64_t currX = std::int64_t(targetX)+entry->i;
        const std::int64_t currY = std::int64_t(targetY)+entry->j;
        const std::int64_t currZ = std::int64_t(targetZ)+entry->k;
        const CellBlock& block = m_cells[size_t(static_cast<unsigned int>(currX))*size_t(m_nCells[1])*size_t(m_nCells[2])+size_t(static_cast<unsigned int>(currY))*size_t(m_nCells[2])+size_t(static_cast<unsigned int>(currZ))];
        if constexpr (rules) {
          if (block.count == 0) continue;
          if (wholeShell) {
            visit.emit(block.start, block.count);
            continue;
          }
          if (verdicts ? visit.settle(faces, currX, currY, currZ, X, Y, Z, block.start, block.count)
                       : cellBeyond(faces, currX, currY, currZ, X, Y, Z, threshold())) continue;
        }
        else if (block.count == 0 || cellBeyond(faces, currX, currY, currZ, X, Y, Z, threshold())) continue;
        const size_t first = block.start;
        scanBlock(m_X.data()+first, m_Y.data()+first, m_Z.data()+first, block.count, X, Y, Z,
                  [&] (const size_t k, const double dist_sq) { visit(first, k, dist_sq); });
      }
    }
  }
}

#endif
