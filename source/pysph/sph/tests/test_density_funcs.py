"""
Tests for the density_funcs module.
"""

# standard imports
import unittest
import numpy
import os

# local imports
import pysph.base.api as base
import pysph.sph.api as sph
import pysph.solver.api as solver

from pysph.sph.funcs.density_funcs import SPHRho, SPHDensityRate
from pysph.base.particle_array import ParticleArray
from pysph.base.kernels import Poly6Kernel, CubicSplineKernel

from function_test_template import FunctionTestCase

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)

    def setup_calcs(self):
        pass

class SummationDensityTestCase(FunctionTestCase):
    """ The setup consists of four particles placed at the
    vertices of a unit square. 
    
    """    
    def setup(self):
        self.func = sph.SPHRho.withargs().get_func(self.pa, self.pa)

    def get_reference_solution(self):
        """ Evaluate the force on each particle manually """
        
        pa = self.pa
        rhos = []

        x,y,z,p,m,h,rho = pa.get('x','y','z','p','m','h','rho')

        kernel = base.CubicSplineKernel(dim=2)

        for i in range(self.np):

            force = base.Point()
            rho = 0.0
            xi, yi, zi = x[i], y[i], z[i]

            ri = base.Point(xi,yi,zi)

            hi = h[i]

            for j in range(self.np):

                grad = base.Point()
                xj, yj, zj = x[j], y[j], z[j]
                hj, mj = m[j], h[j]

                havg = 0.5 * (hi + hj)

                rj = base.Point(xj, yj, zj)
        
                wij = kernel.py_function(ri, rj, havg)

                rho += mj*wij

            force.x = rho
            rhos.append(force)

        return rhos

    def test_single_precision(self):
        self._test('single', nd=6)

    def test_double_precision(self):
        self._test('double', nd=6)

class DensityRateTestCase(FunctionTestCase):
    """ The setup consists of four particles placed at the
    vertices of a unit square. 
    
    """    
    def setup(self):
        self.func = sph.SPHDensityRate.withargs().get_func(self.pa, self.pa)

    def get_reference_solution(self):
        """ Evaluate the force on each particle manually """
        
        pa = self.pa
        rhos = []

        x, y, z, m, h = pa.get('x','y','z','m','h')
        u, v, w = pa.get('u','v','w')

        kernel = base.CubicSplineKernel(dim=2)

        for i in range(self.np):

            rho_sum = 0.0
            force = base.Point()
            xi, yi, zi = x[i], y[i], z[i]

            ri = base.Point(xi,yi,zi)

            hi = h[i]

            for j in range(self.np):

                grad = base.Point()
                xj, yj, zj = x[j], y[j], z[j]
                hj, mj = m[j], h[j]

                vij = base.Point(u[i]-u[j], v[i]-v[j], w[i]-w[j])

                havg = 0.5 * (hi + hj)

                rj = base.Point(xj, yj, zj)
        
                kernel.py_gradient(ri, rj, havg, grad)
                
                rho_sum += mj*vij.dot(grad)

            force.x = rho_sum
            rhos.append(force)

        return rhos

    def test_single_precision(self):
        self._test('single', nd=6)

    def test_double_precision(self):
        self._test('double', nd=7)

if __name__ == '__main__':
    unittest.main()
