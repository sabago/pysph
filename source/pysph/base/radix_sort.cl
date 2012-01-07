/* OpenCL kernels for the RadixSortManager

The RadixSortManager requires cell indices for each particle which are
used as keys for the radix sort. The sorted keys are then examined to
determine the cell counts (indices of particles within each cell)
which can be used to generate a neighbor list for each particle.

The OpenCL kernels for the binning and calculating the cell counts are
defined in this file.

*/

/*
 * @brief Compute the cell indices for each particle based on a cell
 * size and domain limts.
 * @param x                     input buffer of 'x' values
 * @param y                     input buffer of 'y' values
 * @param z                     input buffer of 'z' values
 * @param cellids               outpt buffer of cell indices
 * @param cell_size             cell size to use for binning
 * @param ncx                   number of cells in the 'x' direction
 * @param ncy                   number of cells in the 'y' direction 
 * @param ncz                   number of cells in the 'z' direction
 * @param mcx                   minimim cell index in 'x'
 * @param mcy                   minimim cell index in 'y'
 * @param mcz                   minimim cell index in 'z'

 
 #pragma OPENCL EXTENSION cl_amd_printf : enable
*/
__kernel void bin(__global const REAL* x,
		  __global const REAL* y,
		  __global const REAL* z,
		  __global uint* cellids,
		  REAL const cell_size,
		  int const ncx, 
		  int const ncy,
		  int const ncz,
		  int const mcx,
		  int const mcy, 
		  int const mcz
		  )
{
  // each thread loads one value from global mem given by it's global id
  unsigned int gid = get_global_id(0);

  // compute the cell index for this point
  int IX = (int)floor( x[gid]/cell_size );
  int IY = (int)floor( y[gid]/cell_size );
  int IZ = (int)floor( z[gid]/cell_size );

  // compute the flattened cellid 
  unsigned int cellid = (uint) ((IZ - mcz) * (ncx*ncy) + (IY - mcy)*ncx + (IX - mcx));

  cellids[gid] = cellid;

}

/*
 * @brief Compute the cell counts based on the sorted cell indices.
 * @param cellids                      Input array of sorted cell indices
 * @param cell_counts                  Output array of cell counts
 * @param ncells                       Number of cells
 * @param np                           Number of particles

 This kernel should be launched with as many threads as there are
 particles.

*/
__kernel void compute_cell_counts(__global const uint* cellids, 
				  __global uint* cell_counts,
				  uint const ncells,
				  uint const np)
{

  unsigned int gid = get_global_id(0);
  int cellid;
  int cellidm;

  cellid = cellids[gid];
  
  // Handle the case for the first thread 
  if (gid==0){
    for (int k=0; k<(cellid+1); ++k){
      cell_counts[k] = 0;
    }
  }      

  // Handle the case for the last thread 
  else if (gid==(np-1)){
    for (int k=(cellid+1); k<(ncells+1); ++k){
      cell_counts[k] = np;
    }
    cellidm = cellids[gid-1];
    for (int k=0; k<(cellid-cellidm); ++k){
      cell_counts[cellid-k] = gid;
    }
  }
  
  // Handle all other cases 
  else{
    cellidm = cellids[gid-1];
    for (int k=0; k<(cellid-cellidm); ++k){
      cell_counts[cellid-k] = gid;
    }    
  }

} // __kernel


