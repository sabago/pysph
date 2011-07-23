""" Test for the OpenCL integrators """

import pysph.solver.api as solver
import pysph.base.api as base
import pysph.sph.api as sph

Fluid = base.ParticleType.Fluid

import numpy
import unittest

if solver.HAS_CL:
    import pyopencl as cl
    from pysph.solver.cl_integrator import CLIntegrator

else:
    try:
        import nose.plugins.skip as skip
        reason = "PyOpenCL not installed"
        raise skip.SkipTest(reason)
    except ImportError:
        pass

CLDomain = base.DomainManagerType
CLLocator = base.OpenCLNeighborLocatorType
CYLoctor = base.NeighborLocatorType

class CLIntegratorTestCase(unittest.TestCase):
    """ Test the CLEulerIntegrator """

    def setUp(self):
        """ The setup consists of two fluid particle arrays, each
        having one particle. The fluids are acted upon by an external
        vector force and gravity.

        Comparison is made with the PySPH integration of the system.
        
        """

        x1 = numpy.array([-0.5,])
        y1 = numpy.array([1.0, ])

        x2 = numpy.array([0.5,])
        y2 = numpy.array([1.0,])

        self.f1 = base.get_particle_array(name="fluid1", x=x1, y=y1)
        self.f2 = base.get_particle_array(name="fluid2", x=x2, y=y2)

        self.particles = base.CLParticles(
            arrays=[self.f1,self.f2],
            domain_manager_type=CLDomain.DomainManager,
            cl_locator_type=CLLocator.AllPairNeighborLocator
            )

        
        self.kernel = kernel = base.CubicSplineKernel(dim=2)

        gravity = solver.SPHIntegration(

            sph.GravityForce.withargs(gy=-10.0), on_types=[Fluid],
            updates=['u','v'], id='gravity'

            )

        force = solver.SPHIntegration(

            sph.GravityForce.withargs(gx = -10.0), on_types=[Fluid],
            updates=['u','v'], id='force'

            )

        position = solver.SPHIntegration(

            sph.PositionStepping, on_types=[Fluid],
            updates=['x','y'], id='step',

            )                    
        
        gravity.calc_type = sph.CLCalc
        force.calc_type = sph.CLCalc
        position.calc_type = sph.CLCalc

        gravity_calcs = gravity.get_calcs(self.particles, kernel)
        force_calcs = force.get_calcs(self.particles, kernel)
        position_calcs = position.get_calcs(self.particles, kernel)

        self.calcs = calcs = []
        calcs.extend(gravity_calcs)
        calcs.extend(force_calcs)
        calcs.extend(position_calcs)

        self.integrator = CLIntegrator(self.particles, calcs)

        self.ctx = ctx = solver.create_some_context()
        self.queue = calcs[0].queue

        self.dt = 0.1
        self.nsteps = 10

    def test_setup_integrator(self):
        """ Test the construction of the integrator """

        integrator = self.integrator
        self.integrator.setup_integrator(self.ctx)
        calcs = integrator.calcs

        self.assertEqual( len(calcs), 6 )
        for calc in calcs:
            self.assertTrue( isinstance(calc, sph.CLCalc) )

        # check that setup_cl has been called for the arrays

        self.assertTrue(self.f1.cl_setup_done)
        self.assertTrue(self.f2.cl_setup_done)

        # check for the additional properties created by the integrator

        for arr in [self.f1, self.f2]:

            # Initial props
            self.assertTrue( arr.properties.has_key('_x0') )
            self.assertTrue( arr.properties.has_key('_y0') )
            self.assertTrue( arr.properties.has_key('_u0') )
            self.assertTrue( arr.properties.has_key('_v0') )
            
            self.assertTrue( arr.cl_properties.has_key('cl__x0') )
            self.assertTrue( arr.cl_properties.has_key('cl__y0') )
            self.assertTrue( arr.cl_properties.has_key('cl__u0') )
            self.assertTrue( arr.cl_properties.has_key('cl__v0') )

        # check for the k1 step props

        arr = self.f1

        self.assertTrue( arr.properties.has_key('_a_x_1') )
        self.assertTrue( arr.properties.has_key('_a_y_1') )

        self.assertTrue( arr.properties.has_key('_a_u_1') )
        self.assertTrue( arr.properties.has_key('_a_v_1') )

        self.assertTrue( arr.cl_properties.has_key('cl__a_x_1') )
        self.assertTrue( arr.cl_properties.has_key('cl__a_y_1') )

        self.assertTrue( arr.cl_properties.has_key('cl__a_u_1') )
        self.assertTrue( arr.cl_properties.has_key('cl__a_v_1') )

class CLEulerIntegratorTestCase(CLIntegratorTestCase):
    """ Test the Euler Integration of the system using OpenCL """

    def reference_euler_solution(self, x, y, u, v):
        """ Get the reference solution:

        X = X + h*dt

        """
        dt = self.dt
        fx = -10.0
        fy = -10.0

        x += u*dt
        y += v*dt

        u += fx*dt
        v += fy*dt

        return x, y, u, v

    def test_integrate(self):

        # set the integrator type
        self.integrator = solver.CLEulerIntegrator(self.particles, self.calcs)
        self.integrator.setup_integrator(self.ctx)

        integrator = self.integrator
        f1 = self.f1
        f2 = self.f2

        nsteps = 100

        for i in range(nsteps):
            integrator.integrate(self.dt)

            f1.read_from_buffer()
            f2.read_from_buffer()

            f1x, f1y, f1u, f1v = self.reference_euler_solution(f1.x, f1.y,
                                                               f1.u, f1.v)

            f2x, f2y, f2u, f2v = self.reference_euler_solution(f2.x, f2.y,
                                                               f2.u, f2.v)

            self.assertAlmostEqual(f1.x, f1x, 8)
            self.assertAlmostEqual(f1.y, f1y, 8)
            self.assertAlmostEqual(f1.u, f1u, 8)
            self.assertAlmostEqual(f1.v, f1v, 8)

            self.assertAlmostEqual(f2.x, f2x, 8)
            self.assertAlmostEqual(f2.y, f2y, 8)
            self.assertAlmostEqual(f2.u, f2u, 8)
            self.assertAlmostEqual(f2.v, f2v, 8)


class NBodyIntegrationTestCase(unittest.TestCase):
    """ Compare the integration of particles in OpenCL with PySPH.

    A system of point masses is integrated with OpenCL and in Cython
    and the positions and velocities checked at each time step.

    To achieve this, two separate solvers are created. A Cython one
    and an OpenCL one. The NBody force and position stepping
    operations are added to the solvers just as in the case of the
    examples.

    The command line option `--cl` to the PySPH application should
    work if this test passes.
    
    """

    def setUp(self):

        self.np = np = 100
        
        x = numpy.random.random(np)
        y = numpy.random.random(np)
        z = numpy.random.random(np)
        m = numpy.ones_like(x)

        cy_pa = base.get_particle_array(name="test", x=x, y=y, z=z, m=m)
        cl_pa = base.get_particle_array(name="test", cl_precision="double",
                                        x=x, y=y, z=z, m=m)

        cy_particles = base.Particles(
            [cy_pa,], locator_type=CYLoctor.NSquareNeighborLocator)

        cl_particles = base.CLParticles( [cl_pa,] )

        cy_solver = solver.Solver(
            dim=3, integrator_type=solver.EulerIntegrator)

        cl_solver = solver.Solver(
            dim=3, integrator_type=solver.EulerIntegrator)

        self.cy_solver = cy_solver
        self.cl_solver = cl_solver

        cy_solver.add_operation(solver.SPHIntegration(

            sph.NBodyForce.withargs(), on_types=[0], from_types=[0],
            updates=['u','v','w'], id='nbody_force')

                                )

        cy_solver.add_operation_step([0,])

        cl_solver.add_operation(solver.SPHIntegration(

            sph.NBodyForce.withargs(), on_types=[0], from_types=[0],
            updates=['u','v','w'], id='nbody_force')

                                )

        cl_solver.add_operation_step([0,])
        cl_solver.set_cl(True)
        
        cy_solver.setup(cy_particles)
        cl_solver.setup(cl_particles)

    def test_integrate(self):

        nsteps = 100
        dt = 0.01

        props = ['x','y','z','u','v','w']

        step = 0

        cy_integrator = self.cy_solver.integrator
        cl_integrator = self.cl_solver.integrator

        cy_pa = cy_integrator.particles.arrays[0]
        cl_pa = cl_integrator.particles.arrays[0]

        while ( step < nsteps ):

            cl_pa.read_from_buffer()

            for prop in props:
                cy_prop = cy_pa.get(prop)
                cl_prop = cl_pa.get(prop)

                for j in range(self.np):
                    self.assertAlmostEqual( cl_prop[j], cy_prop[j], 10 )

            cy_integrator.integrate(dt)
            cl_integrator.integrate(dt)

            step += 1
            
if __name__ == '__main__':
    unittest.main()
