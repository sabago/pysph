""" Simple motion. """

import numpy
import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph
import pysph.tools.geometry_utils as geom

from random import randint
from numpy import random

np = 512

def create_particles_3d(**kwargs):

    x = random.random(np)
    y = random.random(np)
    z = random.random(np)

    u = random.random(np) * 0
    v = random.random(np) * 0
    w = random.random(np) * 0

    vol_per_particle = numpy.power(1.0/np, 1.0/3.0)
    radius = 2 * vol_per_particle

    print "Using smoothing length: ", radius

    h = numpy.ones_like(x) * radius

    fluid = base.get_particle_array(name="fluid", type=base.Fluid,
                                    x=x, y=y, z=z,
                                    u=u, v=v, w=w,
                                    h=h)

    x, y, z = geom.create_3D_tank(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.025)
    h = numpy.ones_like(x) * 0.2
    wall = base.get_particle_array(name="wall", type=base.Solid,
                                   x=x, y=y, z=z, h=h)

    print "Number of wall particles = ", wall.get_number_of_particles()

    return [fluid, wall]

def create_particles_2d(**kwargs):

    x = random.random(np)
    y = random.random(np)

    u = random.random(np) * 0
    v = random.random(np) * 0

    vol_per_particle = numpy.power(1.0/np, 1.0/2.0)
    radius = 2 * vol_per_particle

    print "Using smoothing length: ", radius

    h = numpy.ones_like(x) * radius

    fluid = base.get_particle_array(name="fluid", type=base.Fluid,
                                    x=x, y=y,
                                    u=u, v=v,
                                    h=h)

    x, y = geom.create_2D_tank(-0.15, -0.15, 1.15, 1.15, 0.025)
    h = numpy.ones_like(x) * 0.1
    wall = base.get_particle_array(name="wall", type=base.Solid,
                                   x=x, y=y, h=h)

    print "Number of wall particles = ", wall.get_number_of_particles()

    return [fluid, wall]
    

app = solver.Application()

s = solver.Solver(dim=2, integrator_type=solver.RK2Integrator)

s.add_operation(solver.SPHIntegration(

    sph.ArtificialPotentialForce.withargs(factorp=200.0, factorm=0.0),
    on_types=[base.Fluid], from_types=[base.Fluid, base.Solid],
    updates=["u","v", "w"],
    id="potential")

                )


s.add_operation(solver.SPHIntegration(

    sph.PositionStepping.withargs(),
    on_types=[base.Fluid],
    updates=["x","y","z"],
    id="step")

                )

app.setup(
    solver=s,
    variable_h=False,
    create_particles=create_particles_2d)

s.set_time_step(1e-5)
s.set_final_time(25)

app.run()
