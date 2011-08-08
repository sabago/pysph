"""Declarations for the Equation of state SPH functions

"""

# Author: Prabhu Ramachandran <prabhu@aero.iitb.ac.in>
# Copyright (c) 2009, Prabhu Ramachandran

#sph imports
from pysph.sph.sph_func cimport SPHFunction

cdef class IdealGasEquation(SPHFunction):
    """ Ideal gas EOS """
			
    cdef public double gamma

cdef class TaitEquation(SPHFunction):
    """ Tait's equation of state """
    
    cdef public double gamma, co, ro
    cdef public double B

cdef class IsothermalEquation(SPHFunction):
    r""" Isothermal equation of state:

    :math:`$p = c_0^2(\rho - \rho_0)$`

    """
    cdef public double co
    cdef public double ro

cdef class MieGruneisenEquation(SPHFunction):
    cdef public double co
    cdef public double ro
    cdef public double gamma
    cdef public double S


