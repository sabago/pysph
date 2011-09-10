""" Simple motion. """

import numpy
import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

from random import randint
from numpy import random

nx = 1 << 5
dx = 0.5/nx

def create_particles_3d(**kwargs):

    x, y, z = numpy.mgrid[0.25:0.75+1e-10:dx,
                          0.25:0.75+1e-10:dx,
                          0.25:0.75+1e-10:dx]


    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    np = len(x)

    u = random.random(np) * 0
    v = random.random(np) * 0
    w = random.random(np) * 0

    m = numpy.ones_like(x) * dx**3

    vol_per_particle = numpy.power(0.5**3/np ,1.0/3.0)
    radius = 2 * vol_per_particle

    print "Using smoothing length: ", radius

    h = numpy.ones_like(x) * radius

    fluid = base.get_particle_array(name="fluid", type=base.Fluid,
                                    x=x, y=y, z=z,
                                    u=u, v=v, w=w,
                                    m=m,h=h)

    return [fluid,]

def create_particles_2d(**kwargs):

    x, y = numpy.mgrid[0.25:0.75+1e-10:dx, 0.25:0.75+1e-10:dx]
    x = x.ravel()
    y = y.ravel()

    np = len(x)
    
    u = numpy.zeros_like(x)
    v = numpy.zeros_like(x)

    m = numpy.ones_like(x) * dx**2

    vol_per_particle = numpy.power(0.5**2/np ,1.0/3.0)
    radius = 2 * vol_per_particle

    print "Using smoothing length: ", radius

    h = numpy.ones_like(x) * radius

    fluid = base.get_particle_array(name="fluid", type=base.Fluid,
                                    x=x, y=y,
                                    u=u, v=v,
                                    m=m,
                                    h=h)

    return [fluid,]

# define an integrator
class CrazyIntegrator(solver.EulerIntegrator):
    """Crazy integrator """

    def step(self, dt):
        """ Step the particle properties. """
        
        # get the current stage of the integration
        k_num = self.cstep

        for array in self.arrays:

            np = array.get_number_of_particles()

            # get the mapping for this array and this stage
            to_step = self.step_props[ array.name ][k_num]

            for prop in to_step:

                initial_prop = to_step[ prop ][0]
                step_prop = to_step[ prop ][1]

                initial_arr = array.get( initial_prop )
                step_arr = array.get( step_prop )

                updated_array = initial_arr + step_arr * dt

                # simply use periodicity for the positions
                if prop in ['x', 'y', 'z']:
                    for i in range(np):
                        xnew = updated_array[i]
                        if xnew > 1:
                            xnew -= 1
                        if xnew < 0:
                            xnew += 1
                        updated_array[i] = xnew                        

                array.set( **{prop:updated_array} )

        # Increment the step by 1
        self.cstep += 1        
        
app = solver.Application()
s = solver.Solver(dim=2, integrator_type=CrazyIntegrator)

# Update the density of the particles
s.add_operation(solver.SPHOperation(

    sph.SPHRho.withargs(), on_types=[base.Fluid], from_types=[base.Fluid],
    updates=["rho"],
    id="sd")
                )

# Compute some interaction between particles
s.add_operation(solver.SPHIntegration(

    sph.ArtificialPotentialForce.withargs(factorp=1.0, factorm=1.0),
    on_types=[base.Fluid], from_types=[base.Fluid, base.Solid],
    updates=["u","v", "w"],
    id="potential")

                )

# step the particles
s.add_operation(solver.SPHIntegration(

    sph.PositionStepping.withargs(),
    on_types=[base.Fluid],
    updates=["x","y","z"],
    id="step")

                )

app.setup(
    solver=s,
    variable_h=False,
    create_particles=create_particles_3d)

cm = s.particles.cell_manager
print "Number of cells, cell size = %d, %g"%(len(cm.cells_dict), cm.cell_size)

s.set_time_step(1e-2)
s.set_final_time(25)

# add a post step function to save the neighbor information every 10
# iterations
#s.post_step_functions.append( solver.SaveCellManagerData(
#    s.pid, path=s.output_directory, count=10) )

app.run()
