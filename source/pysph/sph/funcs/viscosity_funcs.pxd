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

cdef class MonaghanArtificialViscosity(SPHFunctionParticle):
    """ MonaghanArtificialViscosity """
    
    cdef public double gamma
    cdef public double alpha
    cdef public double beta
    cdef public double eta

    cdef DoubleArray d_dt_fac

cdef class MorrisViscosity(SPHFunctionParticle):
    """
    SPH function to compute pressure gradient.
    """
    cdef str mu
    cdef DoubleArray d_mu, s_mu

    cdef DoubleArray d_dt_fac

cdef class MomentumEquationSignalBasedViscosity(SPHFunctionParticle):

    cdef public double K, beta
    cdef DoubleArray d_dt_fac
    
