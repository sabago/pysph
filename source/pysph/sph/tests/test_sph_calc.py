"""
Tests for the sph_calc module.
"""
# standard imports
import unittest
import numpy

import pysph.solver.api as solver
import pysph.base.api as base
import pysph.sph.api as sph

if solver.HAS_CL:
    import pyopencl as cl

Fluid = base.ParticleType.Fluid
Solid = base.ParticleType.Solid

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)

##############################################################################

def test_sph_calc():

    x = numpy.array([0,])
    y = numpy.array([0,])
    z = numpy.array([0,])
    h = numpy.ones_like(x)

    pa = base.get_particle_array(name="test", x=x, y=y, z=z,h=h)
    particles = base.Particles(arrays=[pa,])
    kernel = base.CubicSplineKernel(dim=1)

    vector_force1 = sph.VectorForce.withargs(force=base.Point(1,1,1))
    vector_force2 = sph.VectorForce.withargs(force=base.Point(1,1,1))

    func1 = vector_force1.get_func(pa,pa)
    func2 = vector_force2.get_func(pa,pa)

    calc = sph.SPHCalc(particles=particles, sources=[pa,pa], dest=pa,
                       kernel=kernel, funcs=[func1, func2],
                       updates=['u','v','w'], integrates=True)

    # evaluate the calc. Accelerations are stored in _tmpx, _tmpy and _tmpz

    calc.sph('_tmpx', '_tmpy', '_tmpz')

    tmpx, tmpy, tmpz = pa.get('_tmpx', '_tmpy', '_tmpz')

    # the acceleration should be 2 in each direction

    assert ( abs(tmpx[0] - 2.0) < 1e-16 )
    assert ( abs(tmpy[0] - 2.0) < 1e-16 )
    assert ( abs(tmpz[0] - 2.0) < 1e-16 )


class CLCalcTestCase(unittest.TestCase):
    """ Test for the CLCalc class """

    def setUp(self):

        if not solver.HAS_CL:
            try:
                import nose.plugins.skip as skip
                reason = "PyOpenCL not installed!"
                raise skip.SkipTest(reason)

            except ImportError:
                pass
        
        self.np = np = 101
        self.x = x = numpy.linspace(0, 1, np)
        self.m = m = numpy.ones_like(x) * (x[1] - x[0])
        self.h = h = 2*self.m

        pa = base.get_particle_array(name="test", cl_precision="single",
                                     x=x, m=m, h=h)

        particles = base.CLParticles([pa,])
        kernel = base.CubicSplineKernel(dim=1)

        func = sph.GravityForce.withargs(gx=-1, gy=-1, gz=-1).get_func(pa,pa)

        self.calc = sph.CLCalc(particles, sources=[pa,], dest=pa,
                               updates=['u','v','w'], kernel=kernel,
                               funcs=[func,])

        if solver.HAS_CL:
            self.context = solver.create_some_context()

    def test_constructor(self):

        calc = self.calc
        func = calc.funcs[0]

        self.assertEqual( calc.cl_kernel_function_name, "GravityForce")
        self.assertEqual( calc.tag, "velocity" )

        self.assertEqual( func.kernel, calc.kernel )

    def test_setup_cl(self):

        calc = self.calc
        func = calc.funcs[0]

    def test_cl_reset_output_arrays(self):

        calc = self.calc
        pa = calc.dest

        calc.setup_cl(self.context)

        calc.cl_reset_output_arrays('_tmpx', '_tmpy', '_tmpz')

        pa._tmpx[:] = 1.0

        pa.read_from_buffer()

        rho = pa.get('_tmpx')

        for i in range(self.np):
            self.assertAlmostEqual( rho[i], 0.0, 10 )

    def test_sph(self):
        calc = self.calc
        pa = calc.dest

        calc.setup_cl(self.context)

        pa._tmpx[:] = 1.0

        calc.sph()

        pa.read_from_buffer()

        tmpx, tmpy, tmpz = pa.get('_tmpx', '_tmpy', '_tmpz')
        for i in range(self.np):
            self.assertAlmostEqual( tmpx[i], -1.0, 10)
            self.assertAlmostEqual( tmpy[i], -1.0, 10)
            self.assertAlmostEqual( tmpz[i], -1.0, 10)

        calc.sph('v','w','y')

        pa.read_from_buffer()

        tmpx, tmpy, tmpz = pa.get('v', 'w', 'y')

        for i in range(self.np):
            self.assertAlmostEqual( tmpx[i], -1.0, 10)
            self.assertAlmostEqual( tmpy[i], -1.0, 10)
            self.assertAlmostEqual( tmpz[i], -1.0, 10)


        
if __name__ == '__main__':
    test_sph_calc()
    unittest.main()
