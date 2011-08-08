"""Declarations for the basic SPH functions 

"""
# Author: Prabhu Ramachandran <prabhu@aero.iitb.ac.in>
# Copyright (c) 2009, Prabhu Ramachandran

#sph imports
from pysph.sph.sph_func cimport SPHFunctionParticle

#base imports 
from pysph.base.particle_array cimport ParticleArray
from pysph.base.point cimport cPoint, cPoint_dot, cPoint_norm
from pysph.base.kernels cimport KernelBase
from pysph.base.carray cimport DoubleArray

cdef class SPHPressureGradient(SPHFunctionParticle):
    """
    SPH function to compute pressure gradient.
    """

    # factors to add artificial pressure as suggested by
    # Monaghan ( SPH Without Tension Instability )
    cdef public double deltap
    cdef public double n
    cdef public double eps

    cdef public bint with_correction

cdef class MomentumEquation(SPHFunctionParticle):
    """ Momentum equation """
    
    cdef public double alpha
    cdef public double beta 
    cdef public double eta
    cdef public double gamma

    # factors to add artificial pressure as suggested by
    # Monaghan ( SPH Without Tension Instability )
    cdef public double deltap
    cdef public double n
    cdef public double eps

    cdef public bint with_correction

    cdef DoubleArray d_dt_fac
