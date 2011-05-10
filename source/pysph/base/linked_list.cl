#define FALSE 0
#define TRUE 1

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

/** 
 * @brief Accquire a lock on a global variable
 * @param lock  Pointer to a global integer which serves as a flag variable.

 Among a group of competing work items, only one accquires the lock
 and the others wait indefinitely till the lock is released by that
 work item. 

 **/
void get_lock(__global int* lock)
{
  int occupied = atom_xchg(lock, TRUE);
  while (occupied == TRUE )
    {
      occupied = atom_xchg(lock, TRUE);
    }
}

/** 
 * @brief Release the global lock
 * @param lock  Pointer to a global integer which serves as a flag variable.

 **/
void release_lock(__global int* lock)
{
  int preval = atom_xchg(lock, FALSE);
}


/** 
 * @brief Initialize an integer array
 * @param val  Value to initialize the array with

 The head and next arrays are initialized to -1 to indicate a NULL
 particle index.

 **/
__kernel void initialize(__global int* array, int const val)
{

  unsigned int gid = get_global_id(0);
  array[gid] = val;
}

/** 
 * @brief Perform a bucket sort on the input data
 * @param x             array of floating point values in the range [0,1]
 * @param y             array of floating point values in the range [0,1]
 * @param z             array of floating point values in the range [0,1]
 * @param cellids       output cell indices for each point
 * @param ix            output cell index in the x direction
 * @param iy            output cell index in the x direction
 * @param iz            output cell index in the x direction
 * @param mx            minimum in x
 * @param my            minimum in y
 * @param mz            minimum in z
 * @param ncx           number of cells in the x direction
 * @param ncy           number of cells in the y direction
 * @param ncz           number of cells in the z direction
 * @param bin_size      bin size for sorting
 **/
__kernel void bin(__global const REAL *x,
		  __global const REAL *y,
		  __global const REAL *z,
		  __global uint* cellids,
		  __global uint* ix,
		  __global uint* iy,
		  __global uint* iz,
		  REAL const mx,
		  REAL const my, 
		  REAL const mz,
		  uint const ncx,
		  uint const ncy,
		  uint const ncz,
		  REAL const cell_size
		  )			    
{

  unsigned int gid = get_global_id(0);
  
  REAL cell_size1 = 1.0F/cell_size;
  
  int _ix = (int)( ( x[gid]-mx ) * cell_size1 );
  int _iy = (int)( ( y[gid]-my ) * cell_size1 );
  int _iz = (int)( ( z[gid]-mz ) * cell_size1 );

  ix[gid] = _ix;
  iy[gid] = _iy;
  iz[gid] = _iz;
  
  int cellid = _iz * (ncx*ncy) + _iy * ncx + _ix;

  cellids[gid] = cellid;

}

__kernel void construct_neighbor_list(__global uint* cellids,
				      __global int* head,
				      __global int* next,
				      __global int* locks


/** 
 * @brief Construct the neighbor list for the particles
 * @param cellids   uint array size np. Particle cell id as a result of binning.
 * @param head      int array size ncells. Index of first particle in cell      
 * @param next      int array size np. Index of next particle in cell.
 * @param locks     uint array size ncells. Semaphore to avoid RMW overlap.
 **/				      )
{
  unsigned int gid = get_global_id(0);
  
  get_lock( &locks[ cellids[gid] ] );

  next[gid] = head[ cellids[gid] ];
  head[ cellids[gid] ] = gid;
  
  release_lock( &locks[ cellids[gid] ] );

} //__kernel
