""" Tests for the External Force Functions """

import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

if solver.HAS_CL:
    import pyopencl as cl

from function_test_template import FunctionTestCase
import numpy
import unittest

class NBodyForceTestCase(FunctionTestCase):
    """ Simple test for the NBodyForce """

    def setup(self):
        """ The setup consists of four particles placed at the
        vertices of a unit square. The force function to be tested is:

        ..math::

                f_i = \sum_{j=1}^{4} \frac{m_j}{|x_j - x_i|^3 +
                \eps}(x_j - x_i)

        The mass of each particle is 1

        """
        self.func = sph.NBodyForce.get_func(self.pa, self.pa)
        self.eps = self.func.eps

    def get_reference_solution(self):
        """ Evaluate the force on each particle manually """

        pa = self.pa
        def get_force(i):
            xi = pa.x[i]; yi = pa.y[i]

            force = base.Point()

            for j in range(self.np):
                xj = pa.x[j]; yj = pa.y[j]

                xji = xj - xi; yji = yj - yi
                dist = numpy.sqrt( xji**2 + yji**2 )

                invr = 1.0/(dist + self.eps)
                invr3 = invr * invr * invr
              
                if not ( i == j ):
                    
                    force.x += invr3 * xji
                    force.y += invr3 * yji

            return force

        forces = [get_force(i) for i in range(self.np)]
        return forces

    def test_single_precision(self):
        self._test('single', nd=6)

    def test_double_precision(self):
        self._test('double', nd=10)

if __name__ == '__main__':
    unittest.main()
