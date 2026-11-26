#include "MeshGridWhole.h"


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
