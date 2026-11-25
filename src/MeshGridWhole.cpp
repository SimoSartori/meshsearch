#include "MeshGridWalk.h"

#include <cstddef>
#include <limits>
#include <vector>

namespace {

  const double dense_cells = 4.;

  const double count_gate_dense = 6.;

  const double count_gate = 8.;

  const double close_gate_dense = 12.;

  double wholeFrom (const double gate, const double cellsize)
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


// ===========================================================================


template <bool Exclude>
std::size_t meshsearch::MeshGrid::countRules (const std::array<unsigned int, 3>& target,
                                              const double X, const double Y, const double Z, const size_t self,
                                              const double Rmax, const double Rmin) const
{
  const double Rmax_sq = Rmax*Rmax;
  const double Rmin_sq = Rmin*Rmin;

  std::size_t count = 0;
  walk(target, X, Y, Z, Rmin, [&] { return Rmax_sq; },
       wholeVisit(wholeTest(Rmax, Rmin, m_cellsize),
                  [&] (const size_t first, const size_t k, const double dist_sq) {
                    if ((!Exclude || first+k != self) && dist_sq <= Rmax_sq && dist_sq >= Rmin_sq) ++count;
                  },
                  [&] (const size_t first, const size_t n) {
                    count += (Exclude && self-first < n) ? n-1 : n;
                  }));
  return count;
}


// ===========================================================================


template <bool Exclude>
void meshsearch::MeshGrid::closeRules (std::vector<unsigned int>& out, const std::array<unsigned int, 3>& target,
                                       const double X, const double Y, const double Z, const size_t self,
                                       const double Rmax, const double Rmin) const
{
  const double Rmax_sq = Rmax*Rmax;
  const double Rmin_sq = Rmin*Rmin;

  std::vector<unsigned int> found;
  found.swap(out);
  found.clear();

  walkPlain(target, X, Y, Z, Rmin, [&] { return Rmax_sq; },
            wholeVisit(wholeTest(Rmax, Rmin, m_cellsize),
                       [&] (const size_t first, const size_t k, const double dist_sq) {
                         if ((!Exclude || first+k != self) && dist_sq <= Rmax_sq && dist_sq >= Rmin_sq)
                           found.push_back(m_toPublic[first+k]);
                       },
                       [&] (const size_t first, const size_t count) {
                         const unsigned int* pub = m_toPublic.data()+first;
                         if (Exclude)
                           emitBlock(pub, first, count, self,
                                     [&] (const unsigned int* begin, const unsigned int* end) { found.insert(found.end(), begin, end); });
                         else found.insert(found.end(), pub, pub+count);
                       }));

  out.swap(found);
}


// ===========================================================================


void meshsearch::MeshGrid::closeRulesPeriodic (std::vector<unsigned int>& out, const std::array<unsigned int, 3>& target,
                                               const double X, const double Y, const double Z, const size_t self,
                                               const double Rmax, const double Rmin) const
{
  const double Rmax_sq = Rmax*Rmax;
  const double Rmin_sq = Rmin*Rmin;

  std::vector<unsigned int> found;
  found.swap(out);
  found.clear();

  walkPeriodic(target, X, Y, Z, Rmin, [&] { return Rmax_sq; },
               wholeVisit(wholeTest(Rmax, Rmin, m_cellsize),
                          [&] (const size_t first, const size_t k, const double dist_sq) {
                            if (first+k != self && dist_sq <= Rmax_sq && dist_sq >= Rmin_sq) found.push_back(m_toPublic[first+k]);
                          },
                          [&] (const size_t first, const size_t count) {
                            emitBlock(m_toPublic.data()+first, first, count, self,
                                      [&] (const unsigned int* begin, const unsigned int* end) { found.insert(found.end(), begin, end); });
                          }));

  out.swap(found);
}


// ===========================================================================


void meshsearch::MeshGrid::wholeCount ()
{
  m_occupied = 0;
  for (const CellBlock& block : m_cells)
    if (block.count != 0) ++m_occupied;
  wholeGates(0);
}


// ===========================================================================


void meshsearch::MeshGrid::wholeGates (const std::ptrdiff_t change)
{
  m_occupied = size_t(std::ptrdiff_t(m_occupied)+change);
  const bool dense = double(m_nObjects) >= dense_cells*double(m_occupied);
  m_countFrom = wholeFrom(dense ? count_gate_dense : count_gate, m_cellsize);
  m_closeFrom = wholeFrom(dense ? close_gate_dense : 0., m_cellsize);
}


// ===========================================================================


std::size_t meshsearch::MeshGrid::countWhole (const double X, const double Y, const double Z,
                                              const double Rmax, const double Rmin) const
{
  const auto target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  checkRadii(Rmax, Rmin);
  return countRules<false>(target, X, Y, Z, no_slot, Rmax, Rmin);
}


// ===========================================================================


std::size_t meshsearch::MeshGrid::countWhole (const unsigned int index, const double Rmax, const double Rmin) const
{
  checkIndex(m_alive, index);
  checkRadii(Rmax, Rmin);

  const size_t self = m_toInternal[index];
  const double X = m_X[self];
  const double Y = m_Y[self];
  const double Z = m_Z[self];

  return countRules<true>(locate(m_lims, m_nCells, m_cellsize, X, Y, Z), X, Y, Z, self, Rmax, Rmin);
}


// ===========================================================================


void meshsearch::MeshGrid::closeWhole (const double X, const double Y, const double Z,
                                       const double Rmax, const double Rmin, std::vector<unsigned int>& out) const
{
  const auto target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  checkRadii(Rmax, Rmin);
  if (m_periodic_any) closeRulesPeriodic(out, target, X, Y, Z, no_slot, Rmax, Rmin);
  else closeRules<false>(out, target, X, Y, Z, no_slot, Rmax, Rmin);
}


// ===========================================================================


void meshsearch::MeshGrid::closeWhole (const unsigned int index, const double Rmax, const double Rmin,
                                       std::vector<unsigned int>& out) const
{
  checkIndex(m_alive, index);
  checkRadii(Rmax, Rmin);

  const size_t self = m_toInternal[index];
  const double X = m_X[self];
  const double Y = m_Y[self];
  const double Z = m_Z[self];

  const auto target = locate(m_lims, m_nCells, m_cellsize, X, Y, Z);
  if (m_periodic_any) closeRulesPeriodic(out, target, X, Y, Z, self, Rmax, Rmin);
  else closeRules<true>(out, target, X, Y, Z, self, Rmax, Rmin);
}
