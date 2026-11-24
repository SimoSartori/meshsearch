#ifndef MESHSEARCH_MESHGRID_H
#define MESHSEARCH_MESHGRID_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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


  /**
   *  @brief Uniform cubic mesh over a set of points, for neighbour queries.
   *
   *  Objects are identified by the index of their coordinates as passed to the
   *  constructor. Indices are stable: removeObject leaves an index permanently
   *  invalid, addObject issues a new one, and indices are never reused or
   *  renumbered. Every member taking an index raises IndexError on an index
   *  that is out of range or removed, except isAlive, which answers false.
   *
   *  Const members may be called concurrently on the same grid. Non-const
   *  members may not.
   *
   *  Coordinates, cellsize and radii must all be in the same unit; none is
   *  imposed. Distances are Euclidean, with the minimum image on the periodic
   *  axes, if any: see the constructor.
   *
   *  Any member that allocates may throw std::bad_alloc, which is not
   *  translated into Error. A constructor that throws builds nothing, and
   *  addObject leaves the grid as it was if it throws.
   *
   *  The class is final, and everything below the public members is private.
   *  Copies are deep, apart from a table that depends only on the geometry
   *  and is shared. Moves transfer the storage without copying it; a
   *  moved-from grid may only be destroyed or assigned to.
   *
   *  The order of the objects within one cell, and so of the results of
   *  get_objectsInCell and closeObjects, is unspecified.
   */
  class MeshGrid final {

  public:

    /// Empty grid. Every member raises or returns empty until it is assigned to.
    MeshGrid () = default;

    MeshGrid (const MeshGrid&) = default;
    MeshGrid (MeshGrid&&) noexcept = default;
    MeshGrid& operator= (const MeshGrid&) = default;
    MeshGrid& operator= (MeshGrid&&) noexcept = default;
    ~MeshGrid () = default;

    /// Construction options.
    struct Options {
      /// Box as {{xlo,xhi},{ylo,yhi},{zlo,zhi}}, finite, with lo <= hi on each
      /// axis. If unset, the data range padded by a margin proportional to
      /// it. Required when any axis is periodic.
      std::optional<std::array<std::array<double,2>,3>> limits;
      /// Per axis, whether distances wrap around the box. All false by default.
      std::array<bool,3> periodic{};
    };

    /// Results of a batch of closeObjects queries, in query order.
    struct Batch {
      /// The indices found by every query, concatenated.
      std::vector<unsigned int> indices;
      /// One more entry than queries, starting at 0: query i found
      /// indices[offsets[i]] up to, excluding, indices[offsets[i+1]].
      std::vector<std::uint64_t> offsets;
    };

    /**
     *  @brief Build the grid.
     *  @param X,Y,Z coordinates, all of the same length, copied into the grid
     *  @param cellsize cell side, finite and positive; one to four mean
     *    separations of the data is the usual range, and docs/benchmark.md
     *    measures what it costs. With a periodic axis it is an upper bound:
     *    see below
     *  @param opt box and periodic axes; see Options
     *  @throw Error if the vectors differ in length; if cellsize is not finite
     *    and positive, or so small that 1/cellsize overflows a double; if
     *    opt.limits is not finite or has lo > hi on an axis; if a coordinate
     *    is not finite or lies outside the limits; if the square of the
     *    grown box's diagonal overflows a double; if the grid would need
     *    more than 2147483647 cells along an axis, or more cells or memory
     *    than 64-bit sizes can count; or if there are more than 4294967295
     *    objects
     *  @throw Error on a periodic axis if opt.limits is unset; if a period
     *    hi-lo is below 2^-1000 (zero extent included) or if
     *    |lo|+|hi|+4*(hi-lo) exceeds 2^1000; or if a period is not an integer
     *    multiple of the shortest one, within a relative 1e-12
     *
     *  Each non-periodic side of the box is grown symmetrically to a whole
     *  number of cells. The grown box contains the limits; where rounding would
     *  leave it inside them, one more cell is added. get_lims returns the grown
     *  box, and members validate points against it, so a point outside the
     *  limits may still be accepted.
     *
     *  Distances are compared through their squares, computed in double
     *  precision as @c dx*dx+dy*dy+dz*dz. @c closeObjects returns every
     *  object whose squared distance lies between @c Rmin*Rmin and
     *  @c Rmax*Rmax, both included; the nearest-neighbour queries rank
     *  objects by the same quantity, ties broken by index. The result
     *  never depends on how objects fall into cells.
     *
     *  An axis of zero extent, where every coordinate is equal or lo == hi,
     *  gets one cell, and the box is grown symmetrically around it to one
     *  cellsize. This does not apply to a periodic axis, which needs a period.
     *
     *  A periodic axis has period L = hi-lo, computed in double, and its box
     *  is exactly the limits: it is not grown and not padded, and coordinates
     *  are not wrapped, so a point outside the limits is an error and a point
     *  at lo and one at hi are two distinct places. The displacement along
     *  the axis is dx = y-q, replaced by dx-L if dx >= L/2 and by dx+L if
     *  dx < -L/2, so that each object is seen once, at its nearest image
     *  (a displacement of exactly +L/2 is taken as -L/2). Both corrections
     *  are exact. Every query then uses this distance, and any radius is
     *  valid, +inf included. All periodic axes share one cell side c, no
     *  larger than cellsize and returned by get_cellsize: the shortest
     *  period is split into the smallest whole number of cells that
     *  keeps them within cellsize, and each periodic axis holds a whole
     *  number of them.
     *
     *  Every check is made before anything is allocated.
     */
    MeshGrid (const std::vector<double>& X,
              const std::vector<double>& Y,
              const std::vector<double>& Z,
              double cellsize,
              const Options& opt);

    /// As above with default Options: the data range padded, no periodic axis.
    /// A separate overload, because a nested aggregate with default member
    /// initializers cannot be a default argument inside its own class.
    MeshGrid (const std::vector<double>& X,
              const std::vector<double>& Y,
              const std::vector<double>& Z,
              double cellsize);

    /// Number of objects in the grid, excluding removed ones.
    unsigned int get_nObjects () const;

    /// True if @p index names an object still in the grid; false if it is out
    /// of range or removed. The one member taking an index that does not raise.
    bool isAlive (unsigned int index) const;

    /// isAlive for each index, in order: 1 or 0.
    std::vector<std::uint8_t> isAlive (const std::vector<unsigned int>& indices) const;

    /// As above, into @p out, which is cleared first and keeps its capacity.
    void isAlive (const std::vector<unsigned int>& indices,
                  std::vector<std::uint8_t>& out) const;

    /// True if a query at the point is accepted, that is, if the point lies
    /// in the box: the test every member taking a point applies. False for a
    /// NaN coordinate, and for every point on a default-constructed grid.
    bool contains (double X, double Y, double Z) const;

    /// contains for each point (X[i], Y[i], Z[i]), in order: 1 or 0.
    /// @throw Error if the vectors differ in length
    std::vector<std::uint8_t> contains (const std::vector<double>& X,
                                        const std::vector<double>& Y,
                                        const std::vector<double>& Z) const;

    /// As above, into @p out, which is cleared first and keeps its capacity.
    void contains (const std::vector<double>& X,
                   const std::vector<double>& Y,
                   const std::vector<double>& Z,
                   std::vector<std::uint8_t>& out) const;

    /// The box: grown to whole cells on a non-periodic axis, exactly the
    /// limits given on a periodic one. All zeros on a default-constructed grid.
    std::array<std::array<double,2>,3> get_lims () const;

    /// Cells per axis.
    std::array<unsigned int, 3> get_nCells () const;

    /// Cell side. It is the cellsize given unless an axis is periodic.
    double get_cellsize () const;

    /// Per axis, whether the box is periodic.
    std::array<bool, 3> get_periodic () const;

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
     *  @brief nearestObject for each point (X[i], Y[i], Z[i]), in order.
     *  @throw Error if the vectors differ in length, if a point is outside
     *    the box, or if the grid holds no objects, even for no point
     *
     *  On a throw, the queries already answered are lost. This holds for
     *  every batch member.
     */
    std::vector<unsigned int> nearestObject (const std::vector<double>& X,
                                             const std::vector<double>& Y,
                                             const std::vector<double>& Z) const;

    /// As above, into @p out, the last argument, which is cleared first and
    /// keeps its capacity; its content is unspecified if the call throws.
    /// This holds for every member taking @p out.
    void nearestObject (const std::vector<double>& X,
                        const std::vector<double>& Y,
                        const std::vector<double>& Z,
                        std::vector<unsigned int>& out) const;

    /**
     *  @brief nearestObject for each object of @p indices, in order.
     *  @throw IndexError if an index is out of range or removed
     *  @throw Error if fewer than two objects remain, even for no index
     */
    std::vector<unsigned int> nearestObject (const std::vector<unsigned int>& indices) const;

    /// As above, into @p out, which may be @p indices itself.
    void nearestObject (const std::vector<unsigned int>& indices,
                        std::vector<unsigned int>& out) const;

    /**
     *  @brief The @p N objects nearest to a point, nearest first; ties by
     *    ascending index.
     *  @param N at most get_nObjects(); 0 returns an empty vector
     *  @param X,Y,Z the point, which must lie inside the box
     *  @throw Error if the point is outside the box, or if N is too large
     *
     *  Nothing is excluded: an object lying on the point is at distance 0 and
     *  is among them. The result is the first @p N objects ordered by
     *  (distance, index), so a tie at the last place keeps the lower indices.
     */
    std::vector<unsigned int> nearestObjects (unsigned int N,
                                              double X, double Y, double Z) const;

    /**
     *  @brief The @p N objects nearest to object @p index, which is excluded;
     *    nearest first; ties by ascending index.
     *  @param N at most get_nObjects()-1; 0 returns an empty vector
     *  @param index the object to measure from, itself excluded
     *  @throw IndexError if @p index is out of range or removed
     *  @throw Error if N is too large
     *
     *  @p index is excluded by identity, not by distance: another object
     *  coincident with it is at distance 0 and is among them. The result is
     *  the first @p N objects ordered by (distance, index), so a tie at the
     *  last place keeps the lower indices.
     */
    std::vector<unsigned int> nearestObjects (unsigned int N,
                                              unsigned int index) const;

    /// The point overload of nearestObjects, into @p out, which is cleared
    /// first.
    void nearestObjects (unsigned int N, double X, double Y, double Z,
                         std::vector<unsigned int>& out) const;

    /// The index overload of nearestObjects, into @p out, which is cleared
    /// first.
    void nearestObjects (unsigned int N, unsigned int index,
                         std::vector<unsigned int>& out) const;

    /**
     *  @brief nearestObjects for each point (X[i], Y[i], Z[i]).
     *  @return n*N indices, query i's N at [i*N, (i+1)*N), each in the order
     *    of the single query
     *  @throw Error if the vectors differ in length, if a point is outside
     *    the box, or if N exceeds get_nObjects(), even for no point
     */
    std::vector<unsigned int> nearestObjects (unsigned int N,
                                              const std::vector<double>& X,
                                              const std::vector<double>& Y,
                                              const std::vector<double>& Z) const;

    /// As above, into @p out.
    void nearestObjects (unsigned int N,
                         const std::vector<double>& X,
                         const std::vector<double>& Y,
                         const std::vector<double>& Z,
                         std::vector<unsigned int>& out) const;

    /**
     *  @brief nearestObjects for each object of @p indices.
     *  @return n*N indices, query i's N at [i*N, (i+1)*N)
     *  @throw IndexError if an index is out of range or removed
     *  @throw Error if N is not below get_nObjects(), even for no index
     */
    std::vector<unsigned int> nearestObjects (unsigned int N,
                                              const std::vector<unsigned int>& indices) const;

    /// As above, into @p out, which may be @p indices itself.
    void nearestObjects (unsigned int N, const std::vector<unsigned int>& indices,
                         std::vector<unsigned int>& out) const;

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
     *  A shell reaching past the box is truncated, not an error, except on a
     *  periodic axis, where it wraps around and each object is counted once.
     *  An infinite Rmax is accepted and selects every object from Rmin
     *  outwards.
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

    /// The point overload of closeObjects, into @p out, which is cleared
    /// first; with Rmin, then with Rmin = 0.
    void closeObjects (double X, double Y, double Z, double Rmax, double Rmin,
                       std::vector<unsigned int>& out) const;
    void closeObjects (double X, double Y, double Z, double Rmax,
                       std::vector<unsigned int>& out) const;

    /// The index overload of closeObjects, into @p out, which is cleared
    /// first; with Rmin, then with Rmin = 0.
    void closeObjects (unsigned int index, double Rmax, double Rmin,
                       std::vector<unsigned int>& out) const;
    void closeObjects (unsigned int index, double Rmax,
                       std::vector<unsigned int>& out) const;

    /**
     *  @brief closeObjects for each point (X[i], Y[i], Z[i]), with the same
     *    radii.
     *  @return each query's indices, in the order the single query returns
     *    them
     *  @throw Error if the vectors differ in length, if a point is outside
     *    the box, or on the radii as closeObjects, even for no point
     */
    Batch closeObjects (const std::vector<double>& X,
                        const std::vector<double>& Y,
                        const std::vector<double>& Z,
                        double Rmax, double Rmin = 0.) const;

    /// As above, into @p out; with Rmin, then with Rmin = 0.
    void closeObjects (const std::vector<double>& X,
                       const std::vector<double>& Y,
                       const std::vector<double>& Z,
                       double Rmax, double Rmin, Batch& out) const;
    void closeObjects (const std::vector<double>& X,
                       const std::vector<double>& Y,
                       const std::vector<double>& Z,
                       double Rmax, Batch& out) const;

    /**
     *  @brief closeObjects for each object of @p indices, with the same
     *    radii; each query excludes its own object.
     *  @throw IndexError if an index is out of range or removed
     *  @throw Error on the radii as closeObjects, even for no index
     */
    Batch closeObjects (const std::vector<unsigned int>& indices,
                        double Rmax, double Rmin = 0.) const;

    /// As above, into @p out; with Rmin, then with Rmin = 0.
    void closeObjects (const std::vector<unsigned int>& indices,
                       double Rmax, double Rmin, Batch& out) const;
    void closeObjects (const std::vector<unsigned int>& indices,
                       double Rmax, Batch& out) const;

    /**
     *  @brief Number of objects closeObjects would return for the point.
     *  @throw Error as closeObjects
     */
    std::size_t countObjects (double X, double Y, double Z,
                              double Rmax, double Rmin = 0.) const;

    /**
     *  @brief Number of objects closeObjects would return for object
     *    @p index, which is excluded.
     *  @throw IndexError if @p index is out of range or removed
     *  @throw Error on the radii as closeObjects
     */
    std::size_t countObjects (unsigned int index,
                              double Rmax, double Rmin = 0.) const;

    /// countObjects for each point (X[i], Y[i], Z[i]), in order. Throws as
    /// the closeObjects batch.
    std::vector<std::uint64_t> countObjects (const std::vector<double>& X,
                                             const std::vector<double>& Y,
                                             const std::vector<double>& Z,
                                             double Rmax, double Rmin = 0.) const;

    /// As above, into @p out; with Rmin, then with Rmin = 0.
    void countObjects (const std::vector<double>& X,
                       const std::vector<double>& Y,
                       const std::vector<double>& Z,
                       double Rmax, double Rmin,
                       std::vector<std::uint64_t>& out) const;
    void countObjects (const std::vector<double>& X,
                       const std::vector<double>& Y,
                       const std::vector<double>& Z,
                       double Rmax, std::vector<std::uint64_t>& out) const;

    /// countObjects for each object of @p indices, in order. Throws as the
    /// closeObjects batch.
    std::vector<std::uint64_t> countObjects (const std::vector<unsigned int>& indices,
                                             double Rmax, double Rmin = 0.) const;

    /// As above, into @p out; with Rmin, then with Rmin = 0.
    void countObjects (const std::vector<unsigned int>& indices,
                       double Rmax, double Rmin,
                       std::vector<std::uint64_t>& out) const;
    void countObjects (const std::vector<unsigned int>& indices,
                       double Rmax, std::vector<std::uint64_t>& out) const;

    /**
     *  @brief Linear index of the cell containing a point.
     *  @throw Error if the point is outside the box
     *
     *  A point within rounding of a cell face may be assigned to the cell on
     *  either side of it. Every member assigns a given point to the same
     *  cell, so an object is always among the objects in the cell of its own
     *  coordinates.
     */
    size_t get_cell (double X, double Y, double Z) const;

    /**
     *  @brief Cell coordinates {ix, iy, iz} of the cell containing a point.
     *  @throw Error if the point is outside the box
     */
    std::array<unsigned int, 3> get_cellCoords (double X, double Y, double Z) const;

    /**
     *  @brief Objects in the cell containing a point.
     *  @throw Error if the point is outside the box
     */
    std::vector<unsigned int> get_objectsInCell (double X, double Y, double Z) const;

    /**
     *  @brief Remove an object from the grid.
     *  @throw IndexError if @p index is out of range or already removed
     *
     *  Its coordinates are no longer kept, and its index stays invalid for
     *  the lifetime of the grid. The memory is not returned: the grid does
     *  not shrink.
     */
    void removeObject (unsigned int index);

    /**
     *  @brief Add an object to the grid.
     *  @return its index, one past the highest ever issued
     *  @throw Error if the point is outside the box, or if 4294967295 indices
     *    have already been issued
     *
     *  The box is not refitted. Indices of removed objects are not reused, and
     *  the memory of a grid that has grown is not reclaimed. To add many
     *  objects, build a new grid: it is faster and uses less memory.
     */
    unsigned int addObject (double X, double Y, double Z);

  private:

    /// Cell offset relative to a target cell.
    struct GridOffset {
      int i;  ///< offset along x, in cells
      int j;  ///< offset along y, in cells
      int k;  ///< offset along z, in cells
    };

    /// Slots of one cell: its objects fill @c count slots from @c start, in
    /// insertion order, and spare slots follow up to @c cap.
    struct CellBlock {
      size_t start;        ///< first slot
      unsigned int count;  ///< objects in the cell
      unsigned int cap;    ///< slots reserved for the cell
    };

    /// Coordinates by slot, so that a cell's objects are contiguous. Spare
    /// slots, and slots a block has moved out of, hold no object.
    std::vector<double> m_X, m_Y, m_Z;

    /// Object index of the object in each slot.
    std::vector<unsigned int> m_toPublic;

    /// Slot of each object, by object index; stale once it is removed.
    std::vector<size_t> m_toInternal;

    /// Per object, whether it is still in the grid; indexed by object index.
    std::vector<char> m_alive;

    /// Count of live objects, maintained by removeObject and addObject.
    unsigned int m_nObjects = 0;

    /// Cell side.
    double m_cellsize = 0.;

    /// Cells per axis.
    std::array<unsigned int, 3> m_nCells = {0, 0, 0};

    /// The box, grown to whole cells, as {{xlo,xhi},{ylo,yhi},{zlo,zhi}}.
    std::array<std::array<double,2>,3> m_lims{};

    /// Per axis, whether it is periodic, and whether any is.
    std::array<bool, 3> m_periodic = {false, false, false};
    bool m_periodic_any = false;

    /// Period of each periodic axis (zero on the others), and a rounded-up
    /// bound on its difference from m_nCells times m_cellsize.
    std::array<double, 3> m_period = {0., 0., 0.};
    std::array<double, 3> m_omega = {0., 0., 0.};

    /// Rounded-up bound, in cells, on how far a distance can fall below the
    /// guaranteed distance of its shell, over all axes; and the same per
    /// axis, in length, on the periodic ones.
    double m_slack = 0.;
    std::array<double, 3> m_slack_axis = {0., 0., 0.};

    /// Cell offsets grouped by guaranteed minimum distance, in cells. Depends
    /// only on m_nCells and m_periodic, so copies of the grid share it.
    std::shared_ptr<const std::vector<std::vector<GridOffset>>> m_mask;

    /// Slot block of each cell, addressed by getLinearIndex. A full cell's
    /// block moves to the end with at least twice the room.
    std::vector<CellBlock> m_cells;

    /// Cell coordinates to linear cell index.
    size_t getLinearIndex (unsigned int ix, unsigned int iy, unsigned int iz) const;

    /// Visit every object of the cells that can hold one within the threshold
    /// of the point, from shell @p Rmin on. @p visit gets the first slot of
    /// the cell, the position in it and the squared distance; @p threshold is
    /// read before each shell and cell. walk dispatches to walkPlain or, on a
    /// grid with a periodic axis, to walkPeriodic.
    template <class Threshold, class Visit>
    void walk (const std::array<unsigned int, 3>& target,
               double X, double Y, double Z, double Rmin,
               Threshold&& threshold, Visit&& visit) const;
    template <class Threshold, class Visit>
    void walkPlain (const std::array<unsigned int, 3>& target,
                    double X, double Y, double Z, double Rmin,
                    Threshold&& threshold, Visit&& visit) const;
    template <class Threshold, class Visit>
    void walkPeriodic (const std::array<unsigned int, 3>& target,
                       double X, double Y, double Z, double Rmin,
                       Threshold&& threshold, Visit&& visit) const;

    /// The single queries on a grid with a periodic axis. @p index, when
    /// not null, names the object the query starts from and replaces the
    /// point.
    unsigned int nearestObjectPeriodic (double X, double Y, double Z,
                                        const unsigned int* index) const;
    void nearestObjectsPeriodic (std::vector<unsigned int>& out, unsigned int N,
                                 double X, double Y, double Z,
                                 const unsigned int* index) const;
    void closeObjectsPeriodic (std::vector<unsigned int>& out,
                               double X, double Y, double Z,
                               double Rmax, double Rmin,
                               const unsigned int* index) const;

    /// Linear index of the cell holding object @p index; 0 if it is not
    /// alive, which the query itself then reports.
    size_t objectCell (unsigned int index) const;

    /// The core of countObjects, on a validated query; with @p Exclude,
    /// slot @p self is skipped.
    template <bool Exclude, class Sink>
    void closeScan (const std::array<unsigned int, 3>& target,
                    double X, double Y, double Z, size_t self,
                    double Rmax, double Rmin, Sink&& sink) const;

  };

}

#endif
