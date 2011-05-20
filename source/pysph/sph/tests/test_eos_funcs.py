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


class TaitEquationTestCase(FunctionTestCase):

    def runTest(self):
        pass

    def setup(self):
        """ The setup consists of four particles placed at the
        vertices of a unit square.

        The function tested is

        ..math::

        `P = B[(\frac{\rho}{\rho0})^gamma - 1.0]`
        cs = c0 * (\frac{\rho}{\rho0})^((gamma-1)/2)


        """
        self.func = sph.TaitEquation.withargs(gamma=7.0).get_func(self.pa,
                                                                  self.pa)

    def get_reference_solution(self):
        """ Evaluate the force on each particle manually """
        
        pa = self.pa
        result = []

        rho = pa.get('rho')

        gamma = 7.0
        co = 1.0
        ro = 1000
        B = co*co*ro/gamma

        for i in range(self.np):

            force = base.Point()

            rhoa = rho[i]

            ratio = rhoa/ro
            gamma2 = 0.5 * (gamma - 1.0)
            tmp = numpy.power(ratio, gamma)

            force.x = (tmp - 1.0) * B
            force.y = numpy.power(ratio, gamma2) * co

            result.append(force)

        return result

    def test_single_precision(self):
        self._test('single', nd=5)

    def test_double_precision(self):
        self._test('double', nd=10)        

if __name__ == '__main__':
    unittest.main()
