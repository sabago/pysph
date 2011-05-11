""" Declarations for the stress SPH functions 

"""

#pysph imports
from pysph.sph.sph_func cimport SPHFunctionParticle, SPHFunction

from pysph.base.particle_array cimport ParticleArray
from pysph.base.kernels cimport KernelBase
from pysph.base.carray cimport DoubleArray

from stress_funcs cimport *

cdef class DamageEquation(StressFunction):
    cdef double alpha, mfac

