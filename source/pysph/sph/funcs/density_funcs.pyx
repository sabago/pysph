from pysph.base.point cimport cPoint_sub, cPoint, cPoint_dot, cPoint_new
from pysph.base.carray cimport DoubleArray
from pysph.base.parray_data cimport ParticleArrayData

from pysph.solver.cl_utils import HAS_CL
if HAS_CL:
    import pyopencl as cl

import numpy


###############################################################################
# `SPHRho` Used for summation density et.al
###############################################################################

# data for the summation density function 
cdef class SPHRhoData(ParticleArrayData):
    pass

# the actual function
cdef class SPHRho(SPHFunctionParticle):
    """ SPH Summation Density """

    def __init__(self, list arrays, list on_types, object dm, list from_types,
                 KernelBase kernel=None, **kwargs):

        SPHFunctionParticle.__init__(self, arrays, on_types, dm, SPHRhoData,
                                     from_types, kernel, **kwargs)

    cdef void eval_self(self, size_t dst_id, object dst_indices):

        cdef size_t n = len(dst_indices)
        cdef KernelBase kernel = self.kernel

        cdef size_t i, j, a, b

        cdef ParticleArrayData d_data = self.dst_data[dst_id]

        # variables used for the summation density operation
        cdef cPoint xa, xb

        cdef double ma, ha
        cdef double w, hab

        for i in range(n):
            a = dst_indices[i]

            ha = d_data.h[a]; ma = d_data.m[a]
            xa = cPoint_new( d_data.x[a], d_data.y[a], d_data.z[a] )

            for j in range(i, n):
                b = dst_indices[j]

                xb = cPoint_new( d_data.x[b], d_data.y[b], d_data.z[b] )
                hab = 0.5 * (ha + d_data.h[b])

                w = kernel.function( xa, xb, hab )

                d_data.rho[ a ] +=  d_data.m[b] * w

                if a != b:
                    d_data.rho[ b ] += ma * w

    cdef void eval_nbr(self, size_t dst_id, size_t src_id,
                       object dst_indices, object src_indices, bint is_symmetric):

        cdef size_t nd = len( dst_indices )
        cdef size_t ns = len( src_indices )

        cdef ParticleArrayData d_data = self.dst_data[ dst_id ]
        cdef ParticleArrayData s_data = self.src_data[ src_id ]

        cdef size_t i, j, a, b
        cdef cPoint xa, xb

        cdef KernelBase kernel = self.kernel

        for i in range(nd):
            a = dst_indices[i]

            ha = d_data.h[ a ]; ma = d_data.m[ a ]
            xa = cPoint_new( d_data.x[a], d_data.y[a], d_data.z[a] )
            
            for j in range(ns):
                b = src_indices[j]

                xb = cPoint_new( d_data.x[b], d_data.y[b], d_data.z[b] )
                hab = 0.5 * (ha + d_data.h[b])

                w = kernel.function( xa, xb, hab )

                d_data.rho[ a ] +=  d_data.m[b] * w
                
                d_data.rho[ b ] += ma * w                
        

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = []

        self.src_reads.extend( ['x','y','z','h','m'] )
        self.dst_reads.extend( ['x','y','z','h','tag'] )

    def _set_extra_cl_args(self):
        pass

    def cl_eval(self, object queue, object context, output1, output2, output3):

        self.set_cl_kernel_args(output1, output2, output3)

        self.cl_program.SPHRho(
            queue, self.global_sizes, self.local_sizes, *self.cl_args).wait()

# ################################################################################
# # `SPHDensityRate` class.
# ################################################################################
# cdef class SPHDensityRate(SPHFunctionParticle):

#     def __init__(self, ParticleArray source, ParticleArray dest,
#                  bint setup_arrays=True, **kwargs):

#         SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True,
#                                      **kwargs)

#         self.id = 'densityrate'
#         self.tag = "density"

#         self.cl_kernel_src_file = "density_funcs.clt"
#         self.cl_kernel_function_name = "SPHDensityRate"
#         self.num_outputs = 1

#     def set_src_dst_reads(self):
#         self.src_reads = []
#         self.dst_reads = []

#         self.src_reads.extend( ['x','y','z','h','m'] )
#         self.dst_reads.extend( ['x','y','z','h','tag'] )

#         self.src_reads.extend( ['u','v','w'] )
#         self.dst_reads.extend( ['u','v','w'] )

#     def _set_extra_cl_args(self):
#         pass        

#     cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
#                        KernelBase kernel, double *nr):
#         """ Compute the contribution of particle at source_pid on particle at
#         dest_pid.
#         """

#         cdef cPoint vel, grad, grada, gradb

#         cdef double ha = self.d_h.data[dest_pid]
#         cdef double hb = self.s_h.data[source_pid]

#         cdef double hab = 0.5 * (ha + hb)

#         cdef DoubleArray xgc, ygc, zgc

#         self._src.x = self.s_x.data[source_pid]
#         self._src.y = self.s_y.data[source_pid]
#         self._src.z = self.s_z.data[source_pid]
        
#         self._dst.x = self.d_x.data[dest_pid]
#         self._dst.y = self.d_y.data[dest_pid]
#         self._dst.z = self.d_z.data[dest_pid]
            
#         vel.x = self.d_u.data[dest_pid] - self.s_u.data[source_pid]
#         vel.y = self.d_v.data[dest_pid] - self.s_v.data[source_pid]
#         vel.z = self.d_w.data[dest_pid] - self.s_w.data[source_pid]

#         if self.hks:
#             grada = kernel.gradient(self._dst, self._src, ha)
#             gradb = kernel.gradient(self._dst, self._src, hb)

#             grad.x = (grada.x + gradb.x) * 0.5
#             grad.y = (grada.y + gradb.y) * 0.5
#             grad.z = (grada.z + gradb.z) * 0.5

#             # grad.set((grada.x + gradb.x)*0.5,
#             #          (grada.y + gradb.y)*0.5,
#             #          (grada.z + gradb.z)*0.5)

#         else:            
#             grad = kernel.gradient(self._dst, self._src, hab)

#         if self.rkpm_first_order_correction:
#             pass

#         if self.bonnet_and_lok_correction:
#             self.bonnet_and_lok_gradient_correction(dest_pid, &grad)

#         nr[0] += cPoint_dot(vel, grad)*self.s_m.data[source_pid]

#     def cl_eval(self, object queue, object context, output1, output2, output3):

#         self.set_cl_kernel_args(output1, output2, output3)

#         self.cl_program.SPHDensityRate(
#             queue, self.global_sizes, self.local_sizes, *self.cl_args).wait()

        
# #############################################################################
