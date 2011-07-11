""" Tests for the basic SPH functions """

import unittest
import numpy

# PySPH imports
import pysph.sph.api as sph
import pysph.base.api as base

class VelocityGradientTestCase(unittest.TestCase):
    """ Test the velocity gradient function.

    Particles are created on a regular grid in 3D with the following
    velocity distribution:

    .. math::

    \vec{v} = ( (x+y+z), (x^2 + y^2 + z^2), (sin(x) + cos(y) )

    The gradient matrix should look like:

    1,    1,    1
    2x,   2y,   2z
    3x^2, 3y^2, 3z^2

    This can be validated for interior particles. Particles are
    considered to be interior if their density is near 1

    """

    def setUp(self):
        dx = 0.02
        x,y,z = numpy.mgrid[0:1.0+dx/2:dx,  0:1.0+dx/2:dx,  0:1.0+dx/2:dx]

        x = x.ravel()
        y = y.ravel()
        z = z.ravel()

        h = numpy.ones_like(x) * 2 * dx
        m = numpy.ones_like(x) * dx
    
        u = x + y + z
        v = x*x + y*y + z*z
        w = x**3 + y**3 + z**3

        self.pa = pa = base.get_particle_array(name="test",
                                               cl_precision="single",
                                               x=x, y=y, z=z, u=u, v=v, w=w,
                                               m=m, h=h)

        particles = base.Particles(arrays=[pa,])
        
        self.kernel = kernel = base.CubicSplineKernel(dim=3)
        
        func = sph.VelocityGradient3D.withargs().get_func(pa,pa)
        
        self.calc = calc = sph.SPHCalc(dest=pa,
                                       sources=[pa,],
                                       particles=particles,
                                       kernel=kernel,
                                       funcs=[func,])

        # create the summation density calc to assign interior particles
        func = sph.SPHRho.withargs().get_func(pa,pa)
        self.sd_calc = sph.SPHCalc(dest=pa,
                                   sources=[pa,],
                                   particles=particles,
                                   kernel=kernel,
                                   funcs=[func,],
                                   updates=['rho'])
        
    def test_constructor(self):
        """ Test the constructor for the function.

        The velocity gradient arrays should have been defined for the
        particle array once the function is created.

        """

        pa = self.pa

        tensor_props = ["v_00", "v_01", "v_02",
                        "v_10", "v_11", "v_12",
                        "v_20", "v_21", "v_22"]

        pa_props = pa.properties.keys()

        # ensure that the particle arrays have been created
        for prop in tensor_props:
            self.assertTrue( prop in pa_props )

    def test_eval(self):

        pa = self.pa
        calc = self.calc

        # Do a summation density on the particles
        self.sd_calc.sph("rho")
        x, y, z, rho = pa.get("x", "y", "z", "rho")

        x2 = x*x
        y2 = y*y
        z2 = z*z

        # Evaluate the velocity gradient calc
        calc.sph()
        
        v_00, v_01, v_02 = pa.get("v_00", "v_01", "v_02")
        v_10, v_11, v_12 = pa.get("v_10", "v_11", "v_12")
        v_20, v_21, v_22 = pa.get("v_20", "v_21", "v_22")

        np = pa.get_number_of_particles()
        for i in range(np):
            if abs( rho[i] - 1.0 ) < 1e-10:

                self.assertAlmostEqual(v_00[i], 1.0, 10)
                self.assertAlmostEqual(v_01[i], 1.0, 10)
                self.assertAlmostEqual(v_01[i], 1.0, 10)

                self.assertAlmostEqual(v_10[i], 2*x[i], 10)
                self.assertAlmostEqual(v_11[i], 2*y[i], 10)
                self.assertAlmostEqual(v_12[i], 2*z[i], 10)

                self.assertAlmostEqual(v_20[i], x2[i], 10)
                self.assertAlmostEqual(v_21[i], y2[i], 10)
                self.assertAlmostEqual(v_22[i], z2[i], 10)                

if __name__ == "__main__":
    unittest.main()
