from pysph.sph.sph_func cimport SPHFunction, SPHFunctionParticle
from pysph.base.carray cimport DoubleArray

cdef class ArtificialPotentialForce(SPHFunctionParticle):
    cdef public double factor1, factor2

cdef class ArtificialPositionStepping(SPHFunction):
    pass
