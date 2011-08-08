"""Test integrator setup for various functions.

Since all the evaluations are taken care by the integrator, we need to
ensure that important data structures like the `initial_properties`
and `step_props` are properly initialized for different functions.

In particular, tensor sph evaluations require that upto 9 step
properties may be evaluated at a time by the calc. 

"""

import unittest
import numpy

import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

class TensorCalcIntegratorSetupTestCase(unittest.TestCase):
    """ Test the setup of the integrator for a tensor evaluation for
    an integrating calc.

    """
    def setUp(self):

        # create a dummy particle array
        self.pa = pa = base.get_particle_array(name="test",
                                               cl_precision="single",
                                               type=1)

        # create a simple solver with one operation
        self.s = s = solver.Solver(3, solver.EulerIntegrator)
        
        # add the velocity gradient operation 
        s.add_operation(solver.SPHOperation(

            sph.VelocityGradient3D.withargs(), on_types=[1,],
            from_types=[1,], id="vgrad")

                             )
        # add the stress rate function
        s.add_operation(solver.SPHIntegration(

            sph.HookesDeviatoricStressRate3D.withargs(),
            on_types=[1,], id="stress_rate")

                             )
        particles = base.Particles(arrays=[pa,])
        s.setup(particles)

        self.integrator = s.integrator

    def test_constructor(self):

        pa = self.pa

        stress_props = ["S_00", "S_01", "S_02",
                        "S_10", "S_11", "S_12",
                        "S_20", "S_21", "S_22"]
            
        vgrad_props = ["v_00", "v_01", "v_02",
                        "v_10", "v_11", "v_12",
                        "v_20", "v_21", "v_22"]

        pa_props = pa.properties.keys()

        for prop in (stress_props+vgrad_props):
            self.assertTrue( prop in pa_props )

    def test_integrator_setup(self):

        pa = self.pa
        integrator = self.integrator

        initial_properties = integrator.initial_properties
        step_props = integrator.step_props

        stress_props = ["S_00", "S_01", "S_02",
                        "S_10", "S_11", "S_12",
                        "S_20", "S_21", "S_22"]

        # initial props is per array and per variable to be stepped
        for step_prop in stress_props:
            self.assertEqual( initial_properties[pa.name][step_prop],
                              '_'+step_prop + '0')
            
        # step props is per array, per stage and per variable to be stepped
        for step_prop in stress_props:
            self.assertEqual( step_props[pa.name][1][step_prop][1],
                              '_a_'+step_prop+'_1')

        # check the calc dst writes for the integrating calc
        for calc in integrator.calcs:
            if calc.integrates:
                # make sure it's the right calc
                self.assertEqual( calc.id, "stress_rate_test" )

                # check the nine components that need to be written to
                writes_ = ['_a_S_00_1', '_a_S_01_1', '_a_S_02_1',
                           '_a_S_10_1', '_a_S_11_1', '_a_S_12_1',
                           '_a_S_20_1', '_a_S_21_1', '_a_S_22_1']

                self.assertEqual( calc.dst_writes[1], writes_ )

if __name__ == "__main__":
    unittest.main()
