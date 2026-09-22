#ifndef MESHSEARCH_MESHGRID_H
#define MESHSEARCH_MESHGRID_H

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace meshsearch {

  /// Thrown on invalid input or on a point outside the box.
  class Error : public std::runtime_error {
  public:
    /// @param what the message, returned by what()
    explicit Error (const std::string& what);
  };


  /// Thrown on an index that is out of range or names a removed object.
  class IndexError : public Error {
  public:
    /// @param what the message, returned by what()
    explicit IndexError (const std::string& what);
  };


  /// Cell offset relative to a target cell.
  struct GridOffset {
    int i;  ///< offset along x, in cells
    int j;  ///< offset along y, in cells
    int k;  ///< offset along z, in cells
  };


  /**
   *  @brief Uniform cubic mesh over a set of points, for neighbour queries.
   *
   *  Objects are identified by the index of their coordinates as passed to the
   *  constructor. Indices are stable: removeObject leaves an index permanently
   *  invalid, addObject issues a new one, and indices are never reused or
   *  renumbered. Naming a removed index raises.
   *
   *  Const members may be called concurrently on the same grid. Non-const
   *  members may not.
   *
   *  Coordinates, cellsize and radii must all be in the same unit; none is
   *  imposed.
   */
  class MeshGrid {

  public:

    /// Empty grid. Every member raises or returns empty until it is assigned to.
    MeshGrid () = default;

    ~MeshGrid () = default;

    /**
     *  @brief Build the grid.
     *  @param X,Y,Z coordinates, all of the same length, copied into the grid
     *  @param cellsize cell side; one to four mean separations of the data is
     *    the usual range, and docs/benchmark.md measures what it costs
     *  @param limits box as {{xlo,xhi},{ylo,yhi},{zlo,zhi}}; if empty, the data
     *    range padded by a margin proportional to it
     *  @throw Error if the vectors differ in length, if cellsize is not
     *    positive, if limits is not 3x2, or if a coordinate lies outside limits
     *
     *  Each side of the box is grown symmetrically to a whole number of cells.
     *  get_lims returns the grown box, and members validate points against it,
     *  so a point outside @p limits may still be accepted.
     */
    MeshGrid (const std::vector<double>& X,
              const std::vector<double>& Y,
              const std::vector<double>& Z,
              double cellsize,
              const std::vector<std::vector<double>>& limits = {});

    /// Number of objects in the grid, excluding removed ones.
    unsigned int get_nObjects () const;

    /// True if @p index names an object still in the grid.
    bool isAlive (unsigned int index) const;

    /// The box, grown to whole cells.
    std::vector<std::vector<double>> get_lims () const;

    /// Cells per axis.
    std::array<unsigned int, 3> get_nCells () const;

    /// Cell side.
    double get_cellsize () const;

    /**
     *  @brief Index of the object nearest to a point.
     *  @throw Error if the point is outside the box, or if the grid holds no
     *    objects
     *
     *  Nothing is excluded: an object lying on the point is at distance 0 and
     *  is the answer.
     */
    unsigned int nearestObject (double X, double Y, double Z) const;

    /**
     *  @brief Index of the object nearest to object @p index, which is excluded.
     *  @throw IndexError if @p index is out of range or removed
     *  @throw Error if no other object remains
     *
     *  @p index is excluded by identity, not by distance: another object
     *  coincident with it is at distance 0 and may be the answer.
     */
    unsigned int nearestObject (unsigned int index) const;

    /**
     *  @brief The @p N objects nearest to a point, nearest first.
     *  @param N at most get_nObjects(); 0 returns an empty vector
     *  @param X,Y,Z the point, which must lie inside the box
     *  @throw Error if the point is outside the box, or if N is too large
     *
     *  Nothing is excluded: an object lying on the point is at distance 0 and
     *  is among them.
     */
    std::vector<unsigned int> nearestObjects (unsigned int N,
                                              double X, double Y, double Z) const;

    /**
     *  @brief The @p N objects nearest to object @p index, which is excluded.
     *  @param N at most get_nObjects()-1; 0 returns an empty vector
     *  @param index the object to measure from, itself excluded
     *  @throw IndexError if @p index is out of range or removed
     *  @throw Error if N is too large
     *
     *  @p index is excluded by identity, not by distance: another object
     *  coincident with it is at distance 0 and is among them.
     */
    std::vector<unsigned int> nearestObjects (unsigned int N,
                                              unsigned int index) const;

    /**
     *  @brief Objects at distance d from a point, with Rmin <= d <= Rmax.
     *  @param X,Y,Z the point, which must lie inside the box
     *  @param Rmax outer radius, inclusive; +inf is accepted
     *  @param Rmin inner radius, inclusive; 0 by default, giving a ball
     *  @return the indices, in unspecified order
     *  @throw Error if the point is outside the box, if a radius is negative
     *    or NaN, or if Rmin exceeds Rmax
     *
     *  The interval is closed at both ends, so a ball holds every object
     *  within Rmax of the point and an object at exactly either end is
     *  returned. Rmin == Rmax selects the objects at exactly that distance.
     *
     *  Nothing is excluded by identity: an object lying on the point is at
     *  d = 0 and is returned.
     *
     *  A shell reaching past the box is truncated, not an error. An infinite
     *  Rmax is accepted and selects every object from Rmin outwards.
     */
    std::vector<unsigned int> closeObjects (double X, double Y, double Z,
                                            double Rmax, double Rmin = 0.) const;

    /**
     *  @brief Objects at distance d from object @p index, with
     *  Rmin <= d <= Rmax. The object itself is excluded.
     *  @param index the object to measure from, itself excluded
     *  @param Rmax outer radius, inclusive; +inf is accepted
     *  @param Rmin inner radius, inclusive; 0 by default, giving a ball
     *  @return the indices, in unspecified order
     *  @throw IndexError if @p index is out of range or removed
     *  @throw Error if a radius is negative or NaN, or if Rmin exceeds Rmax
     *
     *  The interval is the point overload's, closed at both ends. @p index is
     *  excluded by identity, not by distance, so another object coincident
     *  with it is at d = 0 and is returned.
     *
     *  An infinite Rmax is accepted and selects every object from Rmin
     *  outwards.
     */
    std::vector<unsigned int> closeObjects (unsigned int index,
                                            double Rmax, double Rmin = 0.) const;

    /**
     *  @brief Linear index of the cell containing a point.
     *  @throw Error if the point is outside the box
     */
    size_t get_Cell (double X, double Y, double Z) const;

    /**
     *  @brief Cell coordinates {ix, iy, iz} of the cell containing a point.
     *  @throw Error if the point is outside the box
     */
    std::array<unsigned int, 3> get_CellCoords (double X, double Y, double Z) const;

    /**
     *  @brief Objects in the cell containing a point.
     *  @throw Error if the point is outside the box
     */
    std::vector<unsigned int> get_ObjectsInCell (double X, double Y, double Z) const;

    /**
     *  @brief Remove an object from the grid.
     *  @throw IndexError if @p index is out of range or already removed
     *
     *  Its coordinates are retained but unreachable, and its index stays
     *  invalid for the lifetime of the grid.
     */
    void removeObject (unsigned int index);

    /**
     *  @brief Add an object to the grid.
     *  @return its index, one past the highest ever issued
     *  @throw Error if the point is outside the box
     *
     *  The box is not refitted. Indices of removed objects are not reused, so
     *  their storage is not reclaimed.
     */
    unsigned int addObject (double X, double Y, double Z);

  protected:

    /// Coordinates in cell order, so that a cell's objects are contiguous;
    /// indexed by internal index, not by object index. Entries of removed
    /// objects persist.
    std::vector<double> m_X, m_Y, m_Z;

    /// Object index to internal index, and internal index back to object
    /// index. Only the object index is ever public.
    std::vector<unsigned int> m_toInternal, m_toPublic;

    /// Per object, whether it is still in the grid; indexed by object index.
    std::vector<char> m_alive;

    /// Count of live objects, maintained by removeObject and addObject.
    unsigned int m_nObjects = 0;

    /// Cell side.
    double m_cellsize = 0.;

    /// Cells per axis.
    std::array<unsigned int, 3> m_nCells = {0, 0, 0};

    /// The box, grown to whole cells, as {{xlo,xhi},{ylo,yhi},{zlo,zhi}}.
    std::vector<std::vector<double>> m_lims;

    /// Cell offsets grouped by guaranteed minimum distance, in cells. Depends
    /// only on m_nCells, so copies of the grid share it.
    std::shared_ptr<const std::vector<std::vector<GridOffset>>> m_mask;

    /// Live internal indices per cell, addressed by getLinearIndex. A cell's
    /// entries are a contiguous ascending range until objects are added.
    std::vector<std::vector<unsigned int>> m_grid;

    /// Cell coordinates to linear cell index.
    size_t getLinearIndex (unsigned int ix, unsigned int iy, unsigned int iz) const;

  };

}

#endif
