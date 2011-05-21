__kernel void step_array(__global REAL* a, __global REAL* b, __global REAL* c,
			 REAL const h)
{
  unsigned int gid = get_global_id(0);
  
  c[gid] = a[gid] + b[gid] * h;

}
