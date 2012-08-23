from libcpp.vector cimport vector

from pysph.base.point cimport *
from pysph.base.carray cimport *

from pysph.base.particle_array cimport ParticleArray

# forward declarations
cdef class CellManager
cdef class PeriodicDomain

cdef class DomainLimits

cdef inline int real_to_int(double val, double step)
cdef inline cIntPoint find_cell_id(cPoint pnt, double cell_size)

cdef inline vector[cIntPoint] construct_immediate_neighbor_list(
    cIntPoint cell_id,
    bint include_self=*, int distance=*)

cdef inline bint cell_encloses_sphere(IntPoint id,
                          double cell_size, cPoint pnt, double radius)

cdef class Cell:
    # Member variables.

    cdef public int index
    cdef public IntPoint id
    cdef public double cell_size
    cdef public CellManager cell_manager
    cdef public list arrays_to_bin
    
    cdef public int jump_tolerance
    cdef public list index_lists
    cdef int num_arrays

    # Periodicity and ghost cells
    cdef public DomainLimits domain
    
    # Member functions.
    cpdef int add_particles(self, Cell cell) except -1
    cpdef int update(self, dict data) except -1
    cpdef long get_number_of_particles(self)
    cpdef bint is_empty(self)
    cpdef get_centroid(self, Point pnt)

    cpdef set_cell_manager(self, CellManager cell_manager)
    
    cpdef int clear(self) except -1
    cpdef Cell get_new_sibling(self, IntPoint id)
    cpdef get_particle_ids(self, list particle_id_list)
    cpdef get_particle_counts_ids(self, list particle_list,
                                  LongArray particle_counts)

    cpdef insert_particles(self, int parray_id, LongArray indices)
    cpdef clear_indices(self, int parray_id)

    cdef _init_index_lists(self)
    cpdef set_cell_manager(self, CellManager cell_manager)
    cpdef Cell get_new_sibling(self, IntPoint id)
    cpdef Cell copy(self, IntPoint cid, int particle_tag=*)


cdef class CellManager:
    
    # cell size to use for binning
    cdef public double cell_size

    # flag to indicate that particles require to be rebinned
    cdef public bint is_dirty

    # mapping between array and array index in arrays_to_bin
    cdef public dict array_indices

    # the list of particle arrays to bin
    cdef public list arrays_to_bin

    # the number of arrays
    cdef int num_arrays

    # the current Cells used for binning 
    cdef public dict cells_dict

    # optional argumnets to overide the default cell size
    cdef public double min_cell_size, max_cell_size
    cdef public double radius_scale

    # particle jump tolerance in an update
    cdef public int jump_tolerance

    # flag to indicate the manager is initialized
    cdef public bint initialized

    # Periodicity and ghost cells
    cdef public DomainLimits domain
    cdef public bint periodicity
    cdef public dict ghost_cells

    # minimum and maximum smoothing lengths
    cdef double min_h, max_h

    ################################################
    # Member functions
    ################################################

    # Initialize the CellManager
    cpdef initialize(self)

    # Update the CellManager. This re-computes everything from the
    # ghost cells to the physical cells
    cpdef int update(self) except -1

    # Compute the cell size for binning. The default cell size is
    # kernel_radius times the maximum smoothing length.
    cpdef double compute_cell_size(self, double min_size=*, double max_size=*)

    # Build the original cell upon initialization
    cpdef _build_cell(self)

    # Bin particles from the current cell size. This is called from
    # update
    cpdef _rebin_particles(self)

    # Inserts particle indices for a given array to the CellManager
    cpdef _insert_particles(self, int parray_id, LongArray indices)

    # Initialization routine. Generates the mapping between array name
    # to array index.
    cpdef _rebuild_array_indices(self)

    # Set the jump tolerance for all cells. The jump tolerance
    # determines how far a particle may move in a single time step. A
    # particle moving more than the specified tolerance is said to
    # have failed the simulation.
    cpdef set_jump_tolerance(self, int jump_tolerance)

    # Check the jump tolerance for a particle.
    cdef check_jump_tolerance(self, cIntPoint myid, cIntPoint newid)

    # Reset the jump tolerance to 1 for all cells. Called after the
    # initialization is done.
    cdef void _reset_jump_tolerance(self)

    # Remove any unwanted empty cells.
    cpdef list delete_empty_cells(self)

    # Return a new cell with with a specified index
    cpdef Cell get_new_cell(self, IntPoint id)
    
    cdef int get_potential_cells(self, cPoint pnt, double radius,
                                 list cell_list) except -1

    cdef int _get_cells_within_radius(self, cPoint pnt, double radius,
                                      list cell_list) except -1

    # Get the total number of particles
    cpdef long get_number_of_particles(self)

    # Create ghost cells using the periodicity information
    cpdef create_ghost_cells(self)

    # Remove any ghost cells from a previous time step.
    cpdef remove_ghost_particles(self)

    # update the dirty bit for the CellManager
    cpdef int update_status(self, bint variable_h) except -1
    
    # add an array to be binned
    cpdef add_array_to_bin(self, ParticleArray parr)

    # update the cells
    cpdef int cells_update(self) except -1

cdef class PeriodicDomain:
    cdef public double xmin, xmax
    cdef public double ymin, ymax
    cdef public double zmin, zmax

    cdef public double xtranslate
    cdef public double ytranslate
    cdef public double ztranslate

cdef class DomainLimits:
    cdef public double xmin, xmax
    cdef public double ymin, ymax
    cdef public double zmin, zmax

    cdef public double xtranslate
    cdef public double ytranslate
    cdef public double ztranslate

    cdef public int dim
    cdef public bint periodicity
