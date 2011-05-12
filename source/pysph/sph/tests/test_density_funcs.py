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

    def _test_all_pair_double_precision(self):
        """ Test the OpenCL double precision and PySPH solution """

        pa = self.pa

        # set the precision for the particle array
        self.pa.set_cl_precision('double')

        self.cl_particles = base.CLParticles(
            arrays=[self.pa,],
            domain_manager_type=CLDomain.DefaultManager,
            cl_locator_type=CLLocator.AllPairNeighborLocator)

        self.setup_calcs()

        # setup OpenCL
        self.cl_calc.setup_cl(self.ctx)

        # get the reference solution
        reference_solution = self.get_reference_solution()        

        # Evaluate the Cython Calc. default outputs are _tmpx, _tmpy and _tmpz
        self.calc.sph()
        cython_tmpx = pa._tmpx.copy()

        # Evaluate the OpenCL calc. default outputs are the same
        pa._tmpx[:] = -1
        self.cl_calc.sph()
        pa.read_from_buffer()

        opencl_tmpx = pa._tmpx        

        for i in range(self.np):
            self.assertAlmostEqual(reference_solution[i], cython_tmpx[i], 6)
            self.assertAlmostEqual(reference_solution[i], opencl_tmpx[i], 6)

if __name__ == '__main__':
    unittest.main()
