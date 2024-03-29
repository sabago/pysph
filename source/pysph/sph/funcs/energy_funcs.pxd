"""Declarations for the basic SPH functions 

"""

# Author: Prabhu Ramachandran <prabhu@aero.iitb.ac.in>
# Copyright (c) 2009, Prabhu Ramachandran

#sph imports
from pysph.sph.sph_func cimport SPHFunctionParticle

#base imports 
from pysph.base.particle_array cimport ParticleArray
from pysph.base.kernels cimport KernelBase
from pysph.base.carray cimport DoubleArray

cdef class EnergyEquationAVisc(SPHFunctionParticle):
    """
    Class to compute the interaction of a boundary particle on a fluid 
    particle.
    """

    cdef public double beta, alpha, gamma, eta

cdef class EnergyEquationNoVisc(SPHFunctionParticle):
    """
    Class to compute the interaction of a boundary particle on a fluid 
    particle.
    """

    pass


cdef class EnergyEquation(SPHFunctionParticle):
    """ Energy equation for the euler equations """

    cdef double alpha, beta, gamma, eta


cdef class ArtificialHeat(SPHFunctionParticle):
    """ Energy equation for the euler equations """
    
    cdef double eta

    # the array for conduction coefficients
    cdef DoubleArray s_q, d_q

cdef class EnergyEquationWithSignalBasedViscosity(SPHFunctionParticle):
    """ """
    cdef public double K, beta, f
