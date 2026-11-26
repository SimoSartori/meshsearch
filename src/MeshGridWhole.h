#ifndef MESHSEARCH_MESHGRIDWHOLE_H
#define MESHSEARCH_MESHGRIDWHOLE_H

#include "MeshGridWalk.h"

#include <cstddef>
#include <limits>
#include <vector>

namespace {

  const double dense_cells = 4.;

  const double count_gate_dense = 6.;

  const double count_gate = 8.;

  const double close_gate_dense = 12.;

  [[maybe_unused]] double wholeFrom (const double gate, const double cellsize)
  {
    return (gate > 0.) ? gate*cellsize : std::numeric_limits<double>::quiet_NaN();
  }

  struct WholeTest {
    double Rmax;
    double Rmax_sq;
    double Rmin_sq;
    bool any_max;
    bool any_min;
    bool accept;
  };

  WholeTest wholeTest (const double Rmax, const double Rmin, const double cellsize)
  {
    WholeTest w;
    w.Rmax = Rmax;
    w.Rmax_sq = Rmax*Rmax;
    w.Rmin_sq = Rmin*Rmin;
    w.any_max = (w.Rmax_sq == std::numeric_limits<double>::infinity());
    w.any_min = (w.Rmin_sq == 0.);
    const double thick = Rmin+cellsize;
    w.accept = w.any_min || w.any_max || (thick*thick+0.5*cellsize*cellsize)*stop_factor <= w.Rmax_sq;
    return w;
  }

  enum class Verdict { scan, accept, skip };

  Verdict cellVerdict (const CellFaces& faces, const std::int64_t cx, const std::int64_t cy, const std::int64_t cz,
                       const double X, const double Y, const double Z, const WholeTest& w)
  {
    const std::int64_t cell[3] = {cx, cy, cz};
    const double q[3] = {X, Y, Z};
    double near[3], far[3];
    for (int a=0; a<3; ++a) {
      const double lower = faces.lo[a]+double(cell[a])*faces.cellsize;
      const double upper = faces.lo[a]+double(cell[a]+1)*faces.cellsize;
      const double gap = (q[a] < lower) ? lower-q[a] : (q[a] > upper) ? q[a]-upper : 0.;
      const double padded = gap-faces.pad[a];
      near[a] = (padded > 0.) ? padded : 0.;
      const double below = q[a]-lower;
      const double above = upper-q[a];
      far[a] = ((below > above) ? below : above)+faces.pad[a];
    }
    const double near_sq = near[0]*near[0]+near[1]*near[1]+near[2]*near[2];
    if (w.Rmax_sq <= huge_sq && near_sq >= tiny_sq && near_sq*skip_factor > w.Rmax_sq) return Verdict::skip;
    const double far_sq = far[0]*far[0]+far[1]*far[1]+far[2]*far[2];
    const bool inside_max = w.any_max || (far_sq >= tiny_sq && far_sq*pad_factor <= w.Rmax_sq);
    const bool outside_min = w.any_min
      || (near_sq >= tiny_sq && w.Rmin_sq <= huge_sq && near_sq*skip_factor >= w.Rmin_sq);
    if (inside_max && outside_min) return Verdict::accept;
    if (!w.any_min && far_sq >= tiny_sq && far_sq*pad_factor < w.Rmin_sq) return Verdict::skip;
    return Verdict::scan;
  }

  bool clearOfRmin (const double bound, const WholeTest& w)
  {
    return bound >= tiny_sq && w.Rmin_sq <= huge_sq && bound >= w.Rmin_sq;
  }

  template <class Each, class Emit>
  class WholeVisit {
  public:
    using whole_rules = void;

    WholeVisit (const WholeTest& test, Each each, Emit emit)
      : m_test(test), m_each(each), m_emit(emit) {}

    void operator() (const size_t first, const size_t k, const double dist_sq)
    {
      m_each(first, k, dist_sq);
    }

    void emit (const size_t first, const size_t count)
    {
      m_emit(first, count);
    }

    unsigned int shells (const CellFaces& faces, const size_t nShells) const
    {
      if (m_test.any_max) return static_cast<unsigned int>(nShells);
      return firstShell(faces, nShells, m_test.Rmax, m_test.Rmax_sq);
    }

    bool shellClear (const unsigned int shell, const double cellsize, const double slack) const
    {
      if (m_test.any_min) return true;
      if (shell == 0) return false;
      const double cells = double(shell);
      if (!(cells > slack)) return false;
      const double reach = (cells-slack)*cellsize;
      return clearOfRmin((reach*reach)*stop_factor, m_test);
    }

    bool shellVerdicts (const unsigned int shell, const double cellsize, const double slack) const
    {
      const double cells = double(shell)+1.;
      const double far = std::sqrt((shell == 0) ? 0.75 : cells*cells+0.5)-slack;
      const double reach = (far > 0.) ? far*cellsize : 0.;
      const double bound = (reach*reach)*stop_factor;
      const bool accept = m_test.accept && (m_test.any_max || bound <= m_test.Rmax_sq);
      const bool skip = !m_test.any_min && bound < m_test.Rmin_sq;
      return accept || skip;
    }

    bool settle (const CellFaces& faces, const std::int64_t cx, const std::int64_t cy, const std::int64_t cz,
                 const double X, const double Y, const double Z, const size_t first, const size_t count)
    {
      const Verdict verdict = cellVerdict(faces, cx, cy, cz, X, Y, Z, m_test);
      if (verdict == Verdict::accept) m_emit(first, count);
      return verdict != Verdict::scan;
    }

  private:
    WholeTest m_test;
    Each m_each;
    Emit m_emit;
  };

  template <class Each, class Emit>
  WholeVisit<Each, Emit> wholeVisit (const WholeTest& test, Each each, Emit emit)
  {
    return WholeVisit<Each, Emit>(test, each, emit);
  }

  template <class Range>
  void emitBlock (const unsigned int* pub, const size_t first, const size_t count, const size_t self, Range&& range)
  {
    const size_t ks = self-first;
    if (ks < count) {
      range(pub, pub+ks);
      range(pub+ks+1, pub+count);
    }
    else range(pub, pub+count);
  }

}

#endif
