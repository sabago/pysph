"""Utility functions for neighbor searching using the radix sort"""

from linked_list_functions import flatten, unflatten

import numpy

def get_neighbors(cellid, ncx, ncy, ncells, cell_counts, indices):
    """Return all particles in neighboring cells.

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
    ix, iy, iz = unflatten(cellid, ncx, ncy)

    nbrs = numpy.ones(0, dtype=numpy.uint32)

    for i in [ix-1, ix, ix+1]:
        for j in [iy-1, iy, iy+1]:
            for k in [iz-1, iz, iz+1]:
                cid = i + j*ncx + k*ncx*ncy
                if ( (cid >= 0) and (cid < ncells) ):
                    tmp = _cell_neighbors(cid, cell_counts, indices)
                    if tmp.size > 0:
                        nbrs = numpy.concatenate( (nbrs, tmp) )

    return nbrs    

def _cell_neighbors(cellid, cell_counts, indices):
    """Return the particle indices for a given cell."""

    start = cell_counts[cellid]
    end = cell_counts[cellid + 1]

    nnbrs = end - start
    nbrs = numpy.ones( nnbrs, numpy.uint32 )

    for i in range(nnbrs):
        nbrs[i] = indices[ start + i ]
        
    return nbrs
    
