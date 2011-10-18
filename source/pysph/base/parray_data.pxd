from particle_array cimport ParticleArray
from carray cimport DoubleArray, LongArray

cdef class ParticleArrayData:

    # the particle array supplying the carrays
    cdef public ParticleArray pa
    
    # default arrays
    cdef DoubleArray x, y, z
    cdef DoubleArray u, v, w
    cdef DoubleArray h, m, rho
    cdef DoubleArray p, e , cs
