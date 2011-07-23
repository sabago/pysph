"""Test functions."""

import numpy
from pysph.base.point cimport cPoint, cPoint_distance, cPoint_sub, \
     cPoint_new, normalized

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
                 bint setup_arrays=True, double fac=0.0, **kwargs):
        
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)

        self.id = 'artificial_potential'
        self.tag = "velocity"

        self.fac = fac

    def set_src_dst_reads(self):
        pass

    def _set_extra_cl_args(self):
        pass
    
    def cl_eval(self, object queue, object context, output1, output2, output3):
        pass

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid,
                       KernelBase kernel, double *nr):
        cdef double hab, q, w
        cdef cPoint rab

        if self.source.name is not self.dest.name:

            
            self._src.x = self.s_x.data[source_pid]
            self._src.y = self.s_y.data[source_pid]
            self._src.z = self.s_z.data[source_pid]
            
            self._dst.x = self.d_x.data[dest_pid]
            self._dst.y = self.d_y.data[dest_pid]
            self._dst.z = self.d_z.data[dest_pid]

            
            hab = 0.5 * ( self.d_h.data[dest_pid] + \
                          self.s_h.data[source_pid] )
            
	    #a-dest, b-src, rab = rb - ra, vector from a-->b
            rab = cPoint_sub(self._src, self._dst)
            rab = normalized(rab)
	    

            q = cPoint_distance(self._src, self._dst)/hab
            
            w = -100.0 * kernel.function(self._dst, self._src, hab)
            #large distance - attraction
            if q > 1:   
                w = w * -1.0 * 0.01

            nr[0] += rab.x * w
            nr[1] += rab.y * w
            nr[2] += rab.z * w

#############################################################################
# `ArtificialPositionStep` class.
#############################################################################
cdef class ArtificialPositionStep(SPHFunction):

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, **kwargs):

        SPHFunction.__init__(self, source, dest, setup_arrays)

        self.id = 'netpot'
        self.tag = "velocity"


    def set_src_dst_reads(self):
        pass

    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):

        cdef size_t i
        
        # get the tag array pointer
        cdef LongArray tag_arr = self.dest.get_carray('tag')

        self.setup_iter_data()
        cdef size_t np = self.dest.get_number_of_particles()
        cdef size_t a

        cdef DoubleArray u = self.d_u
        cdef DoubleArray v = self.d_v
        cdef DoubleArray w = self.d_w

        cdef cPoint _sum = cPoint_new(0,0,0)

        for a in range(np):
            if tag_arr.data[a] == LocalReal:
                _sum.x = _sum.x + u.data[a]
                _sum.y = _sum.y + v.data[a]
                _sum.z = _sum.z + w.data[a]

        _sum.x = _sum.x/np
        _sum.y = _sum.y/np
        _sum.z = _sum.z/np

        for a in range(np):
            if tag_arr.data[a] == LocalReal:
                output1.data[a] = _sum.x
                output2.data[a] = _sum.y
                output3.data[a] = _sum.z

