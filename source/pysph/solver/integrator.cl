__kernel void step_array(__global REAL* a, __global REAL* b, __global REAL* c,
			 REAL const h)
{
  unsigned int gid = get_global_id(0);
  
  c[gid] = a[gid] + b[gid] * h;

}

__kernel void
set_to_zero(__global REAL* array){

  unsigned int gid = get_global_id(0);
  array[gid] = 0.0F;
}


__kernel void
pc_final_step(__global REAL* current_buffer, 
	      __global REAL* initial_buffer){

  unsigned int gid = get_global_id(0);
  current_buffer[gid] = 2.0F * current_buffer[gid] - initial_buffer[gid];
}
