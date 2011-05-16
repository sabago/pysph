""" Time comparison for the Cython and OpenCL integrators.

We use the NBody integration example as the benchmark. Here, and all
neighbor locator is used. The setup consists of four points at the
vertices of the unit square in 2D.


"""

import numpy
from time import time


import pysph.solver.api as solver
import pysph.base.api as base
import pysph.sph.api as sph

import pyopencl as cl

AllPairLocatorCython = base.NeighborLocatorType.NSquareNeighborLocator
AllPairLocatorOpenCL = base.OpenCLNeighborLocatorType.AllPairNeighborLocator
DefaultManager = base.DomainManagerType.DefaultManager

# constants
tf = 1.0
dt = 0.5
nsteps = tf/dt

# generate the particles
x = numpy.array([0.0, 1.0, 1.0, 0.0])
y = numpy.array([0.0, 0.0, 1.0, 1.0])
z = numpy.zeros_like(x)
h = numpy.ones_like(x)
m = numpy.ones_like(x)

precision = "single"
ctx = solver.create_some_context()

pa1 = base.get_particle_array(name="cython", x=x, y=y, z=z, h=h, m=m)
pa2 = base.get_particle_array(name="opencl", cl_precision=precision,
                              x=x, y=y, z=z, h=h, m=m)

particles1 = base.Particles([pa1,], locator_type=AllPairLocatorCython)
particles2 = base.CLParticles([pa2, ])

kernel = base.CubicSplineKernel(dim=2)


# create the cython solver
solver1 = solver.Solver(dim=2, integrator_type=solver.EulerIntegrator)

solver1.add_operation(solver.SPHIntegration(

    sph.NBodyForce.withargs(), on_types=[0], updates=['u','v'],
    id="force")
                      
                      )

solver1.add_operation_step(types=[0])
solver1.setup_integrator(particles1)
solver1.set_final_time(tf)
solver1.set_time_step(dt)

# create the OpenCL solver
solver2 = solver.Solver(dim=2, integrator_type=solver.EulerIntegrator)

solver2.add_operation(solver.SPHIntegration(

    sph.NBodyForce.withargs(), on_types=[0], updates=['u','v'],
    id="force")
                      
                      )

solver2.add_operation_step(types=[0])

solver2.set_cl(True)
solver2.setup_integrator(particles2)
solver2.set_final_time(tf)
solver2.set_time_step(dt)

t1 = time()
solver1.solve()
cython_time = time() - t1

t1 = time()
solver2.solve()
opencl_time = time() - t1

pa2.read_from_buffer()

diff = sum(abs(pa1.x - pa2.x))
diff += sum(abs(pa1.y - pa2.y))

diff += sum(abs(pa1.u - pa2.u))
diff += sum(abs(pa1.v - pa2.v))

diff /= 4

print "=================================================================="
print "OpenCL execution time = %g s"%opencl_time
print "Cython execution time = %g s"%cython_time
print "Speedup = %g"%(cython_time/opencl_time)

if diff < 1e-6:
    print "Results match"

else:
    print "Results dont match"
