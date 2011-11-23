""" Cython helper functions for the linked list and radix sort based
neighbor locators.

Functions defined in this file can be used to get neighbors from the
LinkedListManager and RadixSortManager domain managers.

Both these domain managers use an underlying cell data structure to
bin the particles. Functions defined herin can be used to query the
neighbors within a particular cell or get all neighbors (in 27
neighboring cells) within an SPH context.

"""
import numpy
cimport numpy

cpdef inline int flatten(int ix, int iy, int iz,
                         int ncx, int ncy, int ncz):
    """ Return the flattened index for the 3 dimensional cell index

    Parameters:
    -----------

    ix, iy, iz : int
        Unflattened 3D cell indices

    ncx, ncy, ncz : int
        Number of cells in each coordiante direction

    """
    return iz * (ncx * ncy) + iy * ncx + ix

cpdef inline tuple unflatten(int cid, int ncx, int ncy):
    """Unflatten a three dimensional cell index.

    Parameters:
    -----------

    cid : int
        The flattened cell index

    ncx, ncy : int
        Number of cells in x and y

    """
    cdef int ix, iy, iz
    cdef int tmp = ncx*ncy
    
    iz = cid / tmp

    cid -= (iz * tmp)
    iy = cid / ncx

    ix = cid - (iy * ncx)

    return ix, iy, iz

cpdef numpy.ndarray brute_force_neighbors(
    float xi, float yi, float zi, int np, float radius,
    numpy.ndarray[ndim=1, dtype=numpy.float32_t] x,
    numpy.ndarray[ndim=1, dtype=numpy.float32_t] y,
    numpy.ndarray[ndim=1, dtype=numpy.float32_t] z):
    """ Perform a brute force neighbor search

    Parameters:
    -----------

    xi, yi, zi : float
        Query point

    np : int
        Number of particles to search for

    radius : float
        Search radius

    x, y, z : array

    Note:
    -----

    This function will only work for single precision floating point
    calculations.

    """
    cdef int j, index
    
    cdef numpy.ndarray[ndim=1, dtype=numpy.int32_t] nbrs
    nbrs = numpy.ones(1, numpy.int32)

    cdef float xj, yj, zj
    cdef float dist2

    cdef float radius2 = radius * radius

    index = 0
    for j in range(np):
        xj = x[j]
        yj = y[j]
        zj = z[j]

        dist2 = (xi-xj)*(xi-xj) + (yi-yj)*(yi-yj) + (zi-zj)*(zi-zj)
        if dist2 < radius2:

            if index == nbrs.size:
                nbrs = numpy.resize(nbrs, nbrs.size + 5)

            nbrs[index] = j
            index += 1

    return nbrs[:index]

cpdef cbin(numpy.ndarray[ndim=1, dtype=numpy.float32_t] x,
           numpy.ndarray[ndim=1, dtype=numpy.float32_t] y,
           numpy.ndarray[ndim=1, dtype=numpy.float32_t] z,
           numpy.ndarray[ndim=1, dtype=numpy.uint32_t] bins,
           numpy.ndarray[ndim=1, dtype=numpy.uint32_t] ix,
           numpy.ndarray[ndim=1, dtype=numpy.uint32_t] iy,
           numpy.ndarray[ndim=1, dtype=numpy.uint32_t] iz,
           numpy.ndarray[ndim=1, dtype=numpy.int32_t] head,
           numpy.ndarray[ndim=1, dtype=numpy.int32_t] next,
           float mx, float my, float mz,
           int ncx, int ncy, int ncz, float cell_size, int np,
           int mcx, int mcy, int mcz):
    """ Bin a set of points

    Parameters:
    -----------

    x, y, z : array
        Input points to bin

    bins, ix, iy, iz : array
        Output indices (flattened and unflattened)

    head : array
        Output head array for the linked list

    next : array
        Output next array for the linked list

    """

    cdef int i, _ix, _iy, _iz, _bin

    for i in range(np):
        _ix = numpy.int32( numpy.floor( x[i]/cell_size ) )
        _iy = numpy.int32( numpy.floor( y[i]/cell_size ) )
        _iz = numpy.int32( numpy.floor( z[i]/cell_size ) )

        ix[i] = (_ix - mcx)
        iy[i] = (_iy - mcy)
        iz[i] = (_iz - mcz)

        _bin = (_iz-mcz) * (ncx*ncy) + (_iy-mcy) * ncx + (_ix-mcx)
        bins[i] = _bin
        
        next[i] = head[_bin]
        head[_bin] = numpy.int32(i)

cpdef numpy.ndarray filter_neighbors(
    float xi, float yi, float zi, float radius,
    numpy.ndarray[ndim=1, dtype=numpy.float32_t] x,
    numpy.ndarray[ndim=1, dtype=numpy.float32_t] y,
    numpy.ndarray[ndim=1, dtype=numpy.float32_t] z,
    numpy.ndarray[ndim=1, dtype=numpy.uint32_t] nbrs):
    """ Return neighbors within a specified radius.

    Neighbors returned from all neighboring 27 cells for a particle
    are filtered. A call to this function should be preceeded with a
    call to `get_neighbors` to get all neighbors which may need
    further filtering.

    Parameters:
    -----------

    i : int
        Destination particle source index

    radius : float
        Search radius

    x,y,z : array
        Source arrays

    nbrs : array
        Indices of all source neighbors with respect to the cell structure.

    """
    cdef int j, s_idx, index
    cdef float xj, yj, zj
    cdef float dist2

    cdef int nnbrs = len(nbrs)
    cdef float radius2 = radius * radius

    cdef numpy.ndarray[ndim=1, dtype=numpy.uint32_t] tmp
    tmp = numpy.ones(0, numpy.uint32)

    index = 0
    for j in range(nnbrs):
        s_idx = nbrs[j]

        xj = x[s_idx]; yj = y[s_idx]; zj = z[s_idx]

        dist2 = (xi-xj)*(xi-xj) + (yi-yj)*(yi-yj) + (zi-zj)*(zi-zj)
        if dist2 < radius2:

            if index == tmp.size:
                tmp = numpy.resize(tmp, tmp.size + 5)

            tmp[index] = s_idx
            index += 1

    return tmp[:index]

###########################################################################
# Linked List Functions

# The following functions are to be used to locate neighbors when
# using the LinkedListManager as the domain manager. From PySPH, these
# functions are mainly called from the neighbor locators (locator.py)
###########################################################################
cpdef numpy.ndarray ll_get_neighbors(
    int cellid, int ix, int iy, int iz,
    int ncx, int ncy, int ncells,
    numpy.ndarray[ndim=1, dtype=numpy.int32_t] head,
    numpy.ndarray[ndim=1, dtype=numpy.int32_t] next):
    """ Return  all neighbors for a point from a linked list.

    Parameters:
    -----------

    cellid : int
        Cell index for the destination point

    ix, iy, iz : int
        Unflattened index for the destination point

    ncx, ncy : int
        Number of cells in the `x` and `y` coordinate direction

    ncells : int
        Total number of cells used in the domain

    head : array
        Source Head array with respect to the binning

    next : array
        Source Next array with respect to the binning

    """    
    cdef numpy.ndarray[ndim=1, dtype=numpy.uint32_t] nbrs, tmp
    cdef int i, j, k, cid

    nbrs = numpy.ones(0, numpy.uint32)

    for i in [ix-1, ix, ix+1]:
        for j in [iy-1, iy, iy+1]:
            for k in [iz-1, iz, iz+1]:
                cid = i + j*ncx + k*ncx*ncy

                if ( (cid >= 0) and (cid < ncells) ):
                    tmp = ll_cell_neighbors(cid, head, next)
                    if tmp.size > 0:
                        nbrs = numpy.resize( nbrs, nbrs.size + tmp.size )
                        nbrs[-tmp.size:] = tmp[:]

    return nbrs

cpdef numpy.ndarray ll_cell_neighbors(
    int cellid,
    numpy.ndarray[ndim=1, dtype=numpy.int32_t] head,
    numpy.ndarray[ndim=1, dtype=numpy.int32_t] next):
    """ Use the linked list data structures to find particles within
    the same cell.

    Parameters:
    -----------

    cellid : int
        Cell index for the destination point

    head : array
        Source Head array with respect to the binning

    next : array
        Source Next array with respect to the binning

    """ 
    cdef int next_id, index
    cdef numpy.ndarray[ndim=1, dtype=numpy.uint32_t] nbrs

    nbrs = numpy.ones(0, numpy.uint32)
    
    next_id = head[cellid]

    index = 0
    while ( next_id != -1 ):

        if index == nbrs.size:
            nbrs = numpy.resize(nbrs, (nbrs.size + 10))
            for i in range(10):
                nbrs[-i - 1] = -1

        nbrs[index] = next_id
        next_id = next[next_id]
        index += 1

    return nbrs[:index]

###########################################################################
# Radix sort functions

# The following functions are to be used to locate neighbors when
# using the RadixSortManager as the domain manager. From PySPH, these
# functions are mainly called from the neighbor locators (locator.py)
###########################################################################
cpdef numpy.ndarray rs_get_neighbors(
    unsigned int cellid, int ncx, int ncy, unsigned int ncells,
    numpy.ndarray[ndim=1, dtype=numpy.uint32_t] cell_counts,
    numpy.ndarray[ndim=1, dtype=numpy.uint32_t] indices):
    """Return particle indices from neighboring cells when using the
    RadixSortManager.

    Parameters:
    -----------

    cellid : int
        The index of the cell for which neighbors are sought.

    ncx,ncy : int
        Number of cells in the 'x' and 'y' direction. This is used to
        flatten and unflatten the cell index.

    ncells : int
        Total number of cells.

    cell_counts : array
        An array that determines the start and end indices for particles.
        This is returned by the radix sort manager.

    indices : array
        Particle indices sorted based on the cell indices. This is
        returned by the radix sort manager.

    """

    # get the unflattened id
    cdef int ix, iy, iz
    ix, iy, iz = unflatten(cellid, ncx, ncy)

    cdef numpy.ndarray[ndim=1, dtype=numpy.uint32_t] nbrs
    nbrs = numpy.ones(0, dtype=numpy.uint32)

    for i in [ix-1, ix, ix+1]:
        for j in [iy-1, iy, iy+1]:
            for k in [iz-1, iz, iz+1]:
                cid = i + j*ncx + k*ncx*ncy
                if ( (cid >= 0) and (cid < ncells) ):
                    tmp = rs_cell_neighbors(cid, cell_counts, indices)
                    if tmp.size > 0:
                        nbrs = numpy.concatenate( (nbrs, tmp) )

    return nbrs

cpdef rs_cell_neighbors(
    int cellid,
    numpy.ndarray[ndim=1, dtype=numpy.uint32_t] cell_counts,
    numpy.ndarray[ndim=1, dtype=numpy.uint32_t] indices):
    """Return the particle indices for a given cell."""

    cdef unsigned int start = cell_counts[cellid]
    cdef unsigned int end = cell_counts[cellid + 1]

    cdef unsigned int nnbrs = end - start

    cdef numpy.ndarray[ndim=1, dtype=numpy.uint32_t] nbrs
    nbrs = numpy.ones( nnbrs, numpy.uint32 )

    for i in range(nnbrs):
        nbrs[i] = indices[ start + i ]
        
    return nbrs
