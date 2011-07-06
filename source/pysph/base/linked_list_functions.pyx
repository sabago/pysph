""" Cython functions to get nearest neighbors from the linked list
data structure

Functions defined in this file can be used to get neighbors from the
linked list structure created from a LinkedListManager like so:

# create the manager
import pysph.base.domain_manager as domain_manager
manager = base.domain_manager.LinkedListManager(arrays=[pa1,pa2])

# update the bin structure
manager.update_status()
manager.update()

# optional: Copy buffer contents if using with OpenCL
manager.enqueue_copy()

# now the head and next arrays for a particle are available as
pa1_head = manager.head[pa1.name]
pa1.next = manager.next[pa1.name]

# the flattened and unflattened cell indices are also available as
pa1.ix = manager.ix[pa1.name]
...

#Now, near particles from pa2 from particle `i` in pa1 may be obtained
nbrs = get_neighbors( cellid=manager.cellid[pa1.name][i],
                      ix=manager.ix[pa1.name][i],
                      iy=manager.iy[pa1.name][i],
                      iz=manager.iz[pa1.name][i],
                      ncx=manager.ncx, ncy=manager.ncy, ncells=manager.ncells,
                      head = manager.head[pa2.name]
                      next = manager.next[pa2.name] )        

"""
import numpy
cimport numpy

cpdef inline int get_cellid(int ix, int iy, int iz,
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

cpdef numpy.ndarray get_neighbors_within_radius(
    int i, float radius,
    numpy.ndarray[ndim=1, dtype=numpy.float32_t] x,
    numpy.ndarray[ndim=1, dtype=numpy.float32_t] y,
    numpy.ndarray[ndim=1, dtype=numpy.float32_t] z,
    numpy.ndarray[ndim=1, dtype=numpy.int32_t] nbrs):
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
    cdef float xi = x[i]
    cdef float yi = y[i]
    cdef float zi = z[i]

    cdef float xj, yj, zj
    cdef int nnbrs = len(nbrs)
    cdef float dist2
    cdef float radius2 = radius * radius

    cdef numpy.ndarray[ndim=1, dtype=numpy.int32_t] tmp
    tmp = numpy.ones(0, numpy.int32)

    index = 0
    for j in range(nnbrs):
        s_idx = nbrs[j]

        xj = x[s_idx]
        yj = y[s_idx]
        zj = z[s_idx]

        dist2 = (xi-xj)*(xi-xj) + (yi-yj)*(yi-yj) + (zi-zj)*(zi-zj)
        if dist2 < radius2:

            if index == tmp.size:
                tmp = numpy.resize(tmp, tmp.size + 5)

            tmp[index] = s_idx
            index += 1

    return tmp[:index]
    
cpdef numpy.ndarray get_neighbors(
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
    cdef numpy.ndarray[ndim=1, dtype=numpy.int32_t] nbrs, tmp
    cdef int i, j, k, cid

    nbrs = numpy.ones(0, numpy.int32)

    for i in [ix-1, ix, ix+1]:
        for j in [iy-1, iy, iy+1]:
            for k in [iz-1, iz, iz+1]:
                cid = i + j*ncx + k*ncx*ncy

                if ( (cid >= 0) and (cid < ncells) ):
                    tmp = cell_neighbors(cid, head, next)
                    if tmp.size > 0:
                        nbrs = numpy.resize( nbrs, nbrs.size + tmp.size )
                        nbrs[-tmp.size:] = tmp[:]

    return nbrs

cpdef numpy.ndarray cell_neighbors(
    int cellid,
    numpy.ndarray[ndim=1, dtype=numpy.int32_t] head,
    numpy.ndarray[ndim=1, dtype=numpy.int32_t] next):
    """ Return indices for points within the same cell.

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
    cdef numpy.ndarray[ndim=1, dtype=numpy.int32_t] nbrs

    nbrs = numpy.ones(0, numpy.int32)
    
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
    
