""" Benchmark example for binning particles in Cython and OpenCL """

import numpy
import numpy.random as random
from time import time

import pysph.base.api as base
import pysph.solver.api as solver

CLDomain = base.DomainManagerType
CLLocator = base.OpenCLNeighborLocatorType

# number of points
np = 2**20

# number of times to bin
nbins = 3

# generate the point set

x = random.random(np)
y = random.random(np)
z = random.random(np)

vol_per_particle = numpy.power(1.0/np, 1.0/3.0)
h = numpy.ones_like(x) * 2 * vol_per_particle

precision = "single"
ctx = solver.create_some_context()

pa = base.get_particle_array(name="test", cl_precision=precision,
                             x=x, y=y, z=z, h=h)

t1 = time()
for i in range(nbins):
    particles =  base.Particles([pa,])
    pa.set_dirty(True)
cython_time = time() - t1


t1 = time()

cl_particles = base.CLParticles(
    arrays=[pa,],
    domain_manager_type=CLDomain.LinkedListManager,
    cl_locator_type=CLLocator.LinkedListSPHNeighborLocator)
cl_particles.setup_cl(ctx)

domain_manager = cl_particles.domain_manager

for i in range(nbins - 1):
    domain_manager.is_dirty = False
    domain_manager.update()

opencl_time = time() - t1

print "================================================================"
print "Binning for %d particles using % s precision"%(np, precision)
print "PyOpenCL time = %g s"%(opencl_time)
print "Cython time = %g s"%(cython_time)
print "Speedup = %g"%(cython_time/opencl_time)
