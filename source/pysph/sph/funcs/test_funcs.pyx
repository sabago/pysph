"""Test functions."""

import numpy
from pysph.base.point cimport cPoint, cPoint_distance, cPoint_sub, \
     cPoint_new, cPoint_length, normalized

from pysph.base.particle_array cimport ParticleArray, LocalReal
from pysph.base.carray cimport DoubleArray, LongArray
from pysph.base.kernels cimport KernelBase

cdef extern from "math.h":
    double fabs(double)

#############################################################################
# `ArtificialPotentialForce` class.
#############################################################################
cdef class ArtificialPotentialForce(SPHFunctionParticle):
    """ Class to compute the gravity force on a particle """ 

    #Defined in the .pxd file
    #cdef double gx, gy, gz

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True, double factorp=1.0,
                 factorm = 1.0, **kwargs):
        
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)

        self.id = 'artificial_potential'
        self.tag = "velocity"

        self.factorp = factorp
        self.factorm = factorm

    def set_src_dst_reads(self):
        pass

    def _set_extra_cl_args(self):
        pass
    
    def cl_eval(self, object queue, object context, output1, output2, output3):
        pass

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid,
                       KernelBase kernel, double *nr):
        cdef double hab, q, w, rab_norm
        cdef cPoint rab

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
        
        hab = 0.5 * ( self.d_h.data[dest_pid] + \
                      self.s_h.data[source_pid] )
            
        rab = cPoint_sub(self._dst, self._src)
        rab_norm = cPoint_length(rab)

        if rab_norm > 1e-15:
            rab = normalized(rab)
	    
        q = cPoint_distance(self._src, self._dst)/hab
        w = kernel.function(self._dst, self._src, hab)

        if q > 0.5:
            w *= self.factorm
        else:
            w *= self.factorp

        nr[0] += rab.x * w
        nr[1] += rab.y * w
        nr[2] += rab.z * w

