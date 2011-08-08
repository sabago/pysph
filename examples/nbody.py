""" NBody Example """

import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph
import numpy

Fluid = base.ParticleType.Fluid

# number of particles, time step and final time
np = 1024

dt = 1e-2
tf = 10.0

nsteps = tf/dt

def get_particles(**kwargs):
    
    x = numpy.random.random(np) * 2.0 - 1.0
    y = numpy.random.random(np) * 2.0 - 1.0
    z = numpy.random.random(np) * 2.0 - 1.0
    u = numpy.random.random(np) * 2.0 - 1.0
    v = numpy.random.random(np) * 2.0 - 1.0
    w = numpy.random.random(np) * 2.0 - 1.0
    m = numpy.random.random(np)*100

    pa = base.get_particle_array(name="test", cl_precision="single",
                                 type=Fluid, x=x, y=y, z=z, m=m, u=u,
                                 v=v, w=w)

    return pa

app = solver.Application()

s = solver.Solver(dim=3,
                  integrator_type=solver.EulerIntegrator)

s.add_operation(solver.SPHIntegration(

                sph.NBodyForce.withargs(),
                on_types=[Fluid], from_types=[Fluid],
                updates=['u','v','w'], id='nbody_force')

                  )

s.add_operation_step([Fluid])

app.setup(
    solver=s,    
    variable_h=False, create_particles=get_particles,
    locator_type=base.NeighborLocatorType.NSquareNeighborLocator,
    cl_locator_type=base.OpenCLNeighborLocatorType.AllPairNeighborLocator,
    domain_manager=base.DomainManager
    )

s.set_final_time(tf)
s.set_time_step(dt)
s.set_print_freq(nsteps + 1)

app.run()
