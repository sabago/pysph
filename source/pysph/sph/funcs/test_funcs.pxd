from pysph.sph.sph_func cimport SPHFunction, SPHFunctionParticle


cdef class ArtificialPotentialForce(SPHFunctionParticle):
    cdef public double factorm, factorp
