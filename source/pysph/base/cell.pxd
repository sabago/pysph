from libcpp.vector cimport vector

from pysph.base.point cimport *
from pysph.base.carray cimport *

from pysph.base.particle_array cimport ParticleArray

# forward declarations
cdef class CellManager
cdef class PeriodicDomain

cdef inline int real_to_int(double val, double step)
cdef inline cIntPoint find_cell_id(cPoint pnt, double cell_size)

cdef inline vector[cIntPoint] construct_immediate_neighbor_list(
    cIntPoint cell_id,
    bint include_self=*, int distance=*)

cdef inline bint cell_encloses_sphere(IntPoint id,
                          double cell_size, cPoint pnt, double radius)

cdef class Cell:
    # Member variables.

    cdef public IntPoint id
    cdef public double cell_size
    cdef public CellManager cell_manager
    cdef public list arrays_to_bin
    
    cdef readonly str coord_x
    cdef readonly str coord_y
    cdef readonly str coord_z

    cdef public int jump_tolerance
    cdef public list index_lists
    cdef int num_arrays

    # Periodicity and ghost cells
    cdef public PeriodicDomain periodic_domain
    
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
    cdef public double max_radius_scale

    # particle jump tolerance in an update
    cdef public int jump_tolerance

    # flag to indicate the manager is initialized
    cdef public bint initialized

    # Periodicity and ghost cells
    cdef public PeriodicDomain periodic_domain
    cdef public dict ghost_cells
    
    cdef public str coord_x, coord_y, coord_z
    cdef double min_h, max_h

    # perform the initial binning
    cpdef initialize(self)

    # perform an update
    cpdef int update(self) except -1

    # update the dirty bit for the CellManager
    cpdef int update_status(self) except -1

    # add an array to be binned
    cpdef add_array_to_bin(self, ParticleArray parr)

    # compute the cell size
    cpdef double compute_cell_size(self, double min_size=*, double max_size=*)

    # update the cells
    cpdef int cells_update(self) except -1

    # build the original cell with all particles
    cpdef _build_cell(self)

    # rebin particles
    cpdef rebin_particles(self)

    # insert particles
    cpdef insert_particles(self, int parray_id, LongArray indices)

    cpdef _rebuild_array_indices(self)
    cpdef set_jump_tolerance(self, int jump_tolerance)
    cdef check_jump_tolerance(self, cIntPoint myid, cIntPoint newid)
    cpdef list delete_empty_cells(self)
    cpdef Cell get_new_cell(self, IntPoint id)
    
    cdef int get_potential_cells(self, cPoint pnt, double radius,
                                 list cell_list) except -1

    cdef int _get_cells_within_radius(self, cPoint pnt, double radius,
                                      list cell_list) except -1
    cdef void _reset_jump_tolerance(self)
    cpdef long get_number_of_particles(self)

    cpdef create_ghost_cells(self)
    cpdef remove_ghost_particles(self)

cdef class PeriodicDomain:
    cdef public double xmin, xmax
    cdef public double ymin, ymax
    cdef public double zmin, zmax

    cdef public double xtranslate
    cdef public double ytranslate
    cdef public double ztranslate
