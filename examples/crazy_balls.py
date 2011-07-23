""" Simple motion. """

import numpy
import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

from random import randint

sin = numpy.sin
cos = numpy.cos
pi = numpy.pi

r = 1.0
theta = numpy.linspace(0, 2*pi, 101)

vel_scale = 0.1

def create_particles_2d(**kwargs):
    
    # create the first ball
    x = sin(theta) 
    y = cos(theta)

    vel = randint(1,5) * vel_scale

    dx = x[1] - x[0]
    h = numpy.ones_like(x)*2*dx
    u = numpy.ones_like(x) * vel
    v = numpy.ones_like(x) * vel

    m = numpy.ones_like(x) * 100
    
    ball1 = base.get_particle_array(name="ball1", type=1, cl_precision="single",
                                    x=x, y=y, m=m, h=h, u=u, v=v)

    x = sin(theta) + 3
    y = cos(theta)

    vel = randint(1,5) * vel_scale

    dx = x[1] - x[0]
    h = numpy.ones_like(x)*2*dx
    u = numpy.ones_like(x) * -vel
    v = numpy.ones_like(x) * vel

    m = numpy.ones_like(x) * 100
    
    ball2 = base.get_particle_array(name="ball2", type=1, cl_precision="single",
                                    x=x, y=y, m=m, h=h, u=u, v=v)


    x = sin(theta) + 3
    y = cos(theta) + 3

    vel = randint(1,5) * vel_scale

    dx = x[1] - x[0]
    h = numpy.ones_like(x)*2*dx
    u = numpy.ones_like(x) * -vel
    v = numpy.ones_like(x) * -vel

    m = numpy.ones_like(x) * 100
    
    ball3 = base.get_particle_array(name="ball3", type=1, cl_precision="single",
                                    x=x, y=y, m=m, h=h, u=u, v=v)


    x = sin(theta)
    y = cos(theta) + 3

    vel = randint(1,5) * vel_scale

    dx = x[1] - x[0]
    h = numpy.ones_like(x)*2*dx
    u = numpy.ones_like(x) * vel
    v = numpy.ones_like(x) * -vel

    m = numpy.ones_like(x) * 100
    
    ball4 = base.get_particle_array(name="ball4", type=1, cl_precision="single",
                                    x=x, y=y, m=m, h=h, u=u, v=v)
    


    return [ball1, ball2, ball3, ball4]


app = solver.Application()

s = solver.Solver(dim=2, integrator_type=solver.RK2Integrator)

s.add_operation(solver.SPHIntegration(

    sph.ArtificialPotentialForce.withargs(),
    on_types=[1,], from_types=[1,],
    updates=["u","v", "w"],
    id="potential")

                )


s.add_operation(solver.SPHIntegration(

    sph.ArtificialPositionStep.withargs(),
    on_types=[1,],
    updates=["x","y","z"],
    id="step")

                )

app.setup(
    solver=s,
    variable_h=False,
    create_particles=create_particles_2d)

s.set_time_step(1e-2)
s.set_final_time(25)

app.run()
