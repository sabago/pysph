import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

if solver.HAS_CL:
    import pyopencl as cl

else:
    try:
        import nose.plugins.skip as skip
        reason = "PyOpenCL not installed"
        raise skip.SkipTest(reason)
    except ImportError:
        pass
    
import numpy
import unittest
from os import path

CLDomain = base.DomainManagerType
CLLocator = base.OpenCLNeighborLocatorType

class FunctionTestCase(unittest.TestCase):
    """ Simple test for the NBodyForce """

    def runTest(self):
        pass

    def setUp(self):
        """ The setup consists of four particles placed at the
        vertices of a unit square. The force function to be tested is:

        ..math::

                f_i = \sum_{j=1}^{4} \frac{m_j}{|x_j - x_i|^3 +
                \eps}(x_j - x_i)

        The mass of each particle is 1

        """
        
        self.np = 4

        # define the particle properties here
        x = numpy.array([0, 0, 1, 1], numpy.float64)
        y = numpy.array([0, 1, 1, 0], numpy.float64)

        z = numpy.zeros_like(x)
        m = numpy.ones_like(x)

        u = numpy.array([1, 0, 0, -1], numpy.float64)
        p = numpy.array([0, 0, 1, 1], numpy.float64)

        self.kernel = base.CubicSplineKernel(dim=2)

        # create a ParticleArray with double precision
        self.pa = pa = base.get_particle_array(name="test", x=x,  y=y, z=z,
                                               m=m, u=u, p=p)

        # create a particles instance 
        self.particles = base.Particles([pa,])

        self.cl_particles = base.CLParticles(
            arrays=[self.pa,],
            domain_manager_type=CLDomain.DomainManager,
            cl_locator_type=CLLocator.AllPairNeighborLocator)

        # define the function here
        #self.func = func = sph.NBodyForce.get_func(pa, pa)
        
        if solver.HAS_CL:
            self.ctx = ctx = solver.create_some_context()
            self.q = q = cl.CommandQueue(ctx)

        self.setup()

    def setup(self):
        pass

    def get_reference_solution(self):
        """ Evaluate the force on each particle manually """

        # Define the reference solution here
        raise NotImplementedError

    def setup_calcs(self):
        pa = self.pa
        
        # create a Cython Calc
        calc = sph.SPHCalc( self.particles, [pa,], pa,
                           self.kernel, [self.func,], ['rho'] )

        self.calc = calc

        # create an OpenCL Calc
        cl_calc = sph.CLCalc( self.cl_particles, [pa,], pa,
                              self.kernel, [self.func,], ['rho'] )

        self.cl_calc = cl_calc    

    def _test(self, precision, nd):
        """ Test the PySPH solution """

        pa = self.pa
        pa.set_cl_precision(precision)

        # setup the calcs 
        self.setup_calcs()

        # setup OpenCL
        self.cl_calc.setup_cl(self.ctx)

        # get the reference solution
        reference_solution = self.get_reference_solution()

        self.calc.sph()

        cython_tmpx = pa._tmpx.copy()
        cython_tmpy = pa._tmpy.copy()
        cython_tmpz = pa._tmpz.copy()

        pa._tmpx[:] = -1
        pa._tmpy[:] = -1
        pa._tmpz[:] = -1

        self.cl_calc.sph()
        pa.read_from_buffer()

        opencl_tmpx = pa._tmpx
        opencl_tmpy = pa._tmpy
        opencl_tmpz = pa._tmpz

        for i in range(self.np):
            self.assertAlmostEqual(reference_solution[i].x, cython_tmpx[i],nd)
            self.assertAlmostEqual(reference_solution[i].y, cython_tmpy[i],nd)
            self.assertAlmostEqual(reference_solution[i].z, cython_tmpz[i],nd)

            self.assertAlmostEqual(reference_solution[i].x, opencl_tmpx[i],nd)
            self.assertAlmostEqual(reference_solution[i].y, opencl_tmpy[i],nd)
            self.assertAlmostEqual(reference_solution[i].z, opencl_tmpz[i],nd)
