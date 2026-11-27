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

  /// Thrown on invalid input: a point outside the box, an invalid argument,
  /// or a query the grid cannot answer.
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
   *  @brief Number of threads a batch query may use: the last argument of
   *    every batch form, after Rmin where there is one.
   *
   *  Results do not depend on it: any number of workers gives the same
   *  indices and counts, in the same order, bit for bit, and the same error
   *  on invalid input, which is checked before any worker starts.
   *
   *  One, the default, runs the batch in the calling thread. -1 means every
   *  core available to the process: on Linux the cores of its CPU affinity
   *  set, which is what a cluster job is given, elsewhere
   *  std::thread::hardware_concurrency(). More workers than cores are
   *  allowed but gain nothing. A batch too small to share uses fewer
   *  threads, or one. Threads are started for each call, the calling thread
   *  being one of them; if one cannot be started, the batch continues on
   *  fewer, with the same results. If one throws, such as std::bad_alloc, the others
   *  stop and the exception reaches the caller.
   *
   *  -1 suits a large batch on an otherwise idle process; one suits a
   *  caller that already runs batches from threads of its own.
   */
  struct Workers {
    /// One worker: the batch runs in the calling thread.
    Workers () = default;
    /// @param count at least 1, or -1 for every core available to the process
    /// @throw Error if @p count is 0 or below -1
    explicit Workers (int count);
    /// The count as given: -1, or at least 1.
    int get_count () const { return m_count; }
  private:
    int m_count = 1;
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
   *  imposed. Every member taking a point, except contains, raises Error if
   *  the point lies outside the box.
   *
   *  Distances are Euclidean, with the minimum image on periodic axes: along
   *  a periodic axis of period L the displacement is taken in [-L/2, L/2),
   *  so each object is seen once, at its nearest image. Distances are
   *  compared through their squares, computed in double precision as
   *  dx*dx+dy*dy+dz*dz. Ties between equal distances are broken by
   *  ascending index. No result depends on how objects fall into cells.
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
   *  The same input (coordinates in the same order, cellsize, options, the
   *  same sequence of addObject and removeObject, the same queries) gives
   *  the same results in the same order, on Linux and macOS, on x86-64
   *  and ARM, with any number of threads calling the grid, and with any
   *  number of Workers in a batch; a copy answers as the original does. Which order the objects take within a cell, and so the
   *  order of the results of closeObjects and get_objectsInCell, is not
   *  part of the interface and may change between versions.
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
      /// axis, containing every coordinate. If unset, the box is fitted to
      /// the data with a small margin, and a query point outside the data's
      /// range may raise Error: set it whenever queries can fall outside the
      /// data. Required when any axis is periodic, or when there are no
      /// points.
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
     *  @param cellsize cell side, finite and positive; on a periodic grid an
     *    upper bound, see get_cellsize
     *  @param opt the box and the periodic axes; see Options
     *  @throw Error if the vectors differ in length or hold more than
     *    4294967295 points; if a coordinate is not finite or lies outside
     *    the limits; if cellsize or the limits are invalid; if there are no
     *    points and no limits; if the grid would be too large to address
     *  @throw Error on a periodic axis, if the limits are unset, if the
     *    period is below 2^-1000 or the limits too large to shift by it, or
     *    if the period is not an integer multiple of the shortest one,
     *    within a relative 1e-12
     *
     *  Each non-periodic side of the box is grown symmetrically to a whole
     *  number of cells, always containing the limits; get_lims returns the
     *  grown box. An axis of zero extent gets one cell. A periodic axis has
     *  period hi-lo and its box is exactly the limits; coordinates are not
     *  wrapped. Every check is made before anything is allocated.
     */
    MeshGrid (const std::vector<double>& X,
              const std::vector<double>& Y,
              const std::vector<double>& Z,
              double cellsize,
              const Options& opt);

    /// As above with default Options: the box fitted to the data, no
    /// periodic axis.
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

    /// Cell side. The cellsize given, unless an axis is periodic: then the
    /// largest side not above it that fits a whole number of times into the
    /// shortest period.
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
     *  @param X,Y,Z the points, all of the same length
     *  @param workers threads the batch may use; see Workers
     *  @throw Error if the vectors differ in length, if a point is outside
     *    the box, or if the grid holds no objects, even for no point
     *
     *  On a throw, the queries already answered are lost. Every batch
     *  member takes @p workers last, one by default, and its results do
     *  not depend on it. Both hold for every batch member.
     */
    std::vector<unsigned int> nearestObject (const std::vector<double>& X,
                                             const std::vector<double>& Y,
                                             const std::vector<double>& Z,
                                             Workers workers = Workers()) const;

    /// As above, into @p out, the last argument, which is cleared first and
    /// keeps its capacity; its content is unspecified if the call throws, and
    /// no other thread may use it during the call. This holds for every
    /// member taking @p out. When a batch of indices is given the
    /// container it is answered into, the answer is the one a separate
    /// container would receive; every index batch honours this.
    void nearestObject (const std::vector<double>& X,
                        const std::vector<double>& Y,
                        const std::vector<double>& Z,
                        std::vector<unsigned int>& out,
                        Workers workers = Workers()) const;

    /**
     *  @brief nearestObject for each object of @p indices, in order.
     *  @throw IndexError if an index is out of range or removed
     *  @throw Error if fewer than two objects remain, even for no index
     */
    std::vector<unsigned int> nearestObject (const std::vector<unsigned int>& indices,
                                             Workers workers = Workers()) const;

    /// As above, into @p out, which may be @p indices itself.
    void nearestObject (const std::vector<unsigned int>& indices,
                        std::vector<unsigned int>& out,
                        Workers workers = Workers()) const;

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
                                              const std::vector<double>& Z,
                                              Workers workers = Workers()) const;

    /// The point batch of nearestObjects, into @p out.
    void nearestObjects (unsigned int N,
                         const std::vector<double>& X,
                         const std::vector<double>& Y,
                         const std::vector<double>& Z,
                         std::vector<unsigned int>& out,
                         Workers workers = Workers()) const;

    /**
     *  @brief nearestObjects for each object of @p indices.
     *  @return n*N indices, query i's N at [i*N, (i+1)*N)
     *  @throw IndexError if an index is out of range or removed
     *  @throw Error if N is not below get_nObjects(), even for no index
     */
    std::vector<unsigned int> nearestObjects (unsigned int N,
                                              const std::vector<unsigned int>& indices,
                                              Workers workers = Workers()) const;

    /// As above, into @p out, which may be @p indices itself.
    void nearestObjects (unsigned int N, const std::vector<unsigned int>& indices,
                         std::vector<unsigned int>& out,
                         Workers workers = Workers()) const;

    /**
     *  @brief Objects at distance d from a point, with Rmin <= d <= Rmax.
     *  @param X,Y,Z the point, which must lie inside the box
     *  @param Rmax outer radius, inclusive; +inf is accepted
     *  @param Rmin inner radius, inclusive; 0 by default, giving a ball
     *  @return the indices; their order is not meaningful (see the class)
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
     *  @return the indices; their order is not meaningful (see the class)
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
                        double Rmax, double Rmin = 0.,
                        Workers workers = Workers()) const;

    /// The point batch of closeObjects, into @p out, whose indices and
    /// offsets are both replaced; with Rmin, then with Rmin = 0.
    void closeObjects (const std::vector<double>& X,
                       const std::vector<double>& Y,
                       const std::vector<double>& Z,
                       double Rmax, double Rmin, Batch& out,
                       Workers workers = Workers()) const;
    void closeObjects (const std::vector<double>& X,
                       const std::vector<double>& Y,
                       const std::vector<double>& Z,
                       double Rmax, Batch& out,
                       Workers workers = Workers()) const;

    /**
     *  @brief closeObjects for each object of @p indices, with the same
     *    radii; each query excludes its own object.
     *  @throw IndexError if an index is out of range or removed
     *  @throw Error on the radii as closeObjects, even for no index
     */
    Batch closeObjects (const std::vector<unsigned int>& indices,
                        double Rmax, double Rmin = 0.,
                        Workers workers = Workers()) const;

    /// The index batch of closeObjects, into @p out, whose indices and
    /// offsets are both replaced; @p indices may be out.indices itself. With
    /// Rmin, then with Rmin = 0.
    void closeObjects (const std::vector<unsigned int>& indices,
                       double Rmax, double Rmin, Batch& out,
                       Workers workers = Workers()) const;
    void closeObjects (const std::vector<unsigned int>& indices,
                       double Rmax, Batch& out,
                       Workers workers = Workers()) const;

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
                                             double Rmax, double Rmin = 0.,
                                             Workers workers = Workers()) const;

    /// The point batch of countObjects, into @p out; with Rmin, then with
    /// Rmin = 0.
    void countObjects (const std::vector<double>& X,
                       const std::vector<double>& Y,
                       const std::vector<double>& Z,
                       double Rmax, double Rmin,
                       std::vector<std::uint64_t>& out,
                       Workers workers = Workers()) const;
    void countObjects (const std::vector<double>& X,
                       const std::vector<double>& Y,
                       const std::vector<double>& Z,
                       double Rmax, std::vector<std::uint64_t>& out,
                       Workers workers = Workers()) const;

    /// countObjects for each object of @p indices, in order. Throws as the
    /// closeObjects batch.
    std::vector<std::uint64_t> countObjects (const std::vector<unsigned int>& indices,
                                             double Rmax, double Rmin = 0.,
                                             Workers workers = Workers()) const;

    /// The index batch of countObjects, into @p out; with Rmin, then with
    /// Rmin = 0.
    void countObjects (const std::vector<unsigned int>& indices,
                       double Rmax, double Rmin,
                       std::vector<std::uint64_t>& out,
                       Workers workers = Workers()) const;
    void countObjects (const std::vector<unsigned int>& indices,
                       double Rmax, std::vector<std::uint64_t>& out,
                       Workers workers = Workers()) const;

    /**
     *  @brief Linear index of the cell containing a point:
     *    (ix*ny+iy)*nz+iz, with {ix,iy,iz} from get_cellCoords and
     *    {nx,ny,nz} from get_nCells.
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

    struct GridOffset {
      int i;
      int j;
      int k;
    };

    struct CellBlock {
      size_t start;
      unsigned int count;
      unsigned int cap;
    };

    std::vector<double> m_X, m_Y, m_Z;

    std::vector<unsigned int> m_toPublic;

    std::vector<size_t> m_toInternal;

    std::vector<char> m_alive;

    unsigned int m_nObjects = 0;

    double m_cellsize = 0.;

    std::array<unsigned int, 3> m_nCells = {0, 0, 0};

    std::array<std::array<double,2>,3> m_lims{};

    std::array<bool, 3> m_periodic = {false, false, false};
    bool m_periodic_any = false;

    std::array<double, 3> m_period = {0., 0., 0.};
    std::array<double, 3> m_omega = {0., 0., 0.};

    double m_slack = 0.;
    std::array<double, 3> m_slack_axis = {0., 0., 0.};

    std::shared_ptr<const std::vector<std::vector<GridOffset>>> m_mask;
    std::shared_ptr<const std::vector<std::vector<size_t>>> m_runs;

    std::vector<CellBlock> m_cells;

    size_t m_occupied = 0;
    double m_countFrom = 0.;
    double m_closeFrom = 0.;

    size_t getLinearIndex (unsigned int ix, unsigned int iy, unsigned int iz) const;

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

    unsigned int nearestObjectPeriodic (double X, double Y, double Z,
                                        const unsigned int* index) const;
    void nearestObjectsPeriodic (std::vector<unsigned int>& out, unsigned int N,
                                 double X, double Y, double Z,
                                 const unsigned int* index) const;
    void closeObjectsPeriodic (std::vector<unsigned int>& out,
                               double X, double Y, double Z,
                               double Rmax, double Rmin,
                               const unsigned int* index) const;

    size_t objectCell (unsigned int index) const;

    void wholeCount ();
    void wholeGates (std::ptrdiff_t change);
    std::size_t countWhole (double X, double Y, double Z, double Rmax, double Rmin) const;
    std::size_t countWhole (unsigned int index, double Rmax, double Rmin) const;
    void closeWhole (double X, double Y, double Z, double Rmax, double Rmin, std::vector<unsigned int>& out) const;
    void closeWhole (unsigned int index, double Rmax, double Rmin, std::vector<unsigned int>& out) const;
    template <bool Exclude>
    std::size_t countRules (const std::array<unsigned int, 3>& target, double X, double Y, double Z, size_t self,
                            double Rmax, double Rmin) const;
    template <bool Exclude>
    void closeRules (std::vector<unsigned int>& out, const std::array<unsigned int, 3>& target,
                     double X, double Y, double Z, size_t self, double Rmax, double Rmin) const;
    void closeRulesPeriodic (std::vector<unsigned int>& out, const std::array<unsigned int, 3>& target,
                             double X, double Y, double Z, size_t self, double Rmax, double Rmin) const;
    void closeAppend (std::vector<unsigned int>& out, double X, double Y, double Z, double Rmax, double Rmin,
                      const unsigned int* index) const;

    struct Grouped;

  };

}

#endif
