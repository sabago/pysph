"""Declarations for the GSPH functions.

"""
# Author: Prabhu Ramachandran <prabhu@aero.iitb.ac.in>
# Copyright (c) 2009, Prabhu Ramachandran

#sph imports
from pysph.sph.sph_func cimport SPHFunctionParticle, SPHFunction

#base imports 
from pysph.base.carray cimport DoubleArray, LongArray

cdef class GSPHMomentumEquation(SPHFunctionParticle):

    cdef double gamma

cdef class GSPHEnergyEquation(SPHFunctionParticle):
    cdef double gamma

    cdef DoubleArray d_ustar, d_vstar, d_wstar

cdef class GSPHPositionStepping(SPHFunction):
    cdef DoubleArray d_ustar, d_vstar, d_wstar
