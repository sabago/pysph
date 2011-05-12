""" Tests for the eos functions """
import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

if solver.HAS_CL:
    import pyopencl as cl

import numpy
import unittest

from function_test_template import FunctionTestCase

class IdealGasEquationTestCase(FunctionTestCase):

    def runTest(self):
        pass

    def setup(self):
        """ The setup consists of four particles placed at the
        vertices of a unit square.

        The function tested is

        ..math::

        p_a = (\gamma - 1.0)\rho_a U_a
        cs_a = \sqrt( (\gamma - 1.0) U_a )


        """
        self.func = sph.IdealGasEquation.withargs(gamma=1.4).get_func(self.pa,
                                                                      self.pa)

    def get_reference_solution(self):
        """ Evaluate the force on each particle manually """
        
        pa = self.pa
        result = []

        rho, e = pa.get('rho', 'e')

        kernel = base.CubicSplineKernel(dim=2)
        gamma = 1.4

        for i in range(self.np):

            force = base.Point()

            rhoa = rho[i]
            ea = e[i]

            force.x = (gamma - 1.0) * ea * rhoa
            force.y = numpy.sqrt( (gamma-1.0) * ea )

            result.append(force)

        return result

    def test_single_precision(self):
        self._test('single', nd=6)

    def test_double_precision(self):
        self._test('double', nd=10)

if __name__ == '__main__':
    unittest.main()
