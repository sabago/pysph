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


class XSPHCorrectionTestCase(FunctionTestCase):
    """ The setup consists of four particles placed at the
    vertices of a unit square. 
    
    """    
    def setup(self):
        self.func = sph.XSPHCorrection.withargs().get_func(self.pa, self.pa)

    def get_reference_solution(self):
        """ Evaluate the force on each particle manually """
        
        pa = self.pa
        rhos = []

        x, y, z, m, h, rho = pa.get('x','y','z','m','h', 'rho')
        u, v, w = pa.get('u','v','w')

        eps = 0.5

        kernel = base.CubicSplineKernel(dim=2)

        for i in range(self.np):

            val = 0.0
            force = base.Point()
            xi, yi, zi = x[i], y[i], z[i]

            ri = base.Point(xi,yi,zi)

            hi = h[i]

            for j in range(self.np):

                xj, yj, zj = x[j], y[j], z[j]
                hj, mj = h[j], m[j]

                vji = base.Point(u[j]-u[i], v[j]-v[i], w[j]-w[i])
                rhoij = 0.5 * (rho[i] + rho[j])

                havg = 0.5 * (hi + hj)

                rj = base.Point(xj, yj, zj)
        
                wk = kernel.py_function(ri, rj, havg)
               
                val = m[j]/rhoij * wk * eps

                force.x += val * vji.x
                force.y += val * vji.y
                force.z += val * vji.z

            rhos.append(force)

        return rhos

    def test_single_precision(self):
        self._test('single', nd=6)

    def test_double_precision(self):
        self._test('double', nd=7)


if __name__ == '__main__':
    unittest.main() 
