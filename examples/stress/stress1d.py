""" An example solving stress test case """

import numpy
import sys
import os

import pysph.base.api as base
import pysph.solver.api as solver
from pysph.solver.stress_solver import StressSolver, get_particle_array
from pysph.sph.funcs import stress_funcs, arithmetic_funcs
from pysph.sph.api import SPHFunction

app = solver.Application()

app.opt_parse.add_option('--hfac', action='store', dest='hfac', default=None,
                         type='float',
                         help='the smoothing length as a factor of particle spacing')

app.opt_parse.add_option('--N', action='store', dest='N', default=None, type='float',
                         help='number of partitions (num particles=N+1)')


class PrintPos(object):
    ''' print properties of a particle in a column format (gnuplot/np.loadtxt) '''
    def __init__(self, particle_id, props=['x'], filename='stress.dat'):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        self.file = open(filename, 'w')
        self.file.write('i\tt\t'+'\t'.join(props)+'\n')
        self.res = []
        self.props = props
        self.particle_id = particle_id

    def function(self, solver):
        l = [solver.count, solver.t]
        for prop in self.props:
            l.append(getattr(solver.particles.arrays[0], prop)[self.particle_id])
        self.res.append(l)
        
        s = '\n'.join('\t'.join(map(str,line)) for line in self.res)
        self.file.write(s)
        self.file.write('\n')
        self.res = []


def create_particles():
    N = app.options.N or 20
    N += 1
    hfac = app.options.hfac or 1.2

    rho0 = 1.0
    #x,y = numpy.mgrid[-1.05:1.05+1e-4:dx, -0.105:0.105+1e-4:dx]
    x = numpy.mgrid[0:1:1j*N]
    dx = 1.0/(N-1)
    x = x.ravel()
    #y = y.ravel()
    bdry = (x<=0)
    print bdry, numpy.flatnonzero(bdry)
    m = rho0*numpy.ones_like(x)*dx
    h = numpy.ones_like(x)*hfac*dx
    rho = rho0*numpy.ones_like(x)
    y = z = numpy.zeros_like(x)

    p = z

    #cs = numpy.ones_like(x) * 10000.0

    u = -x
    u *= 0.1

    pa = get_particle_array(x=x, y=y, m=m, rho=rho, h=h, p=p, u=u, v=z, z=z,w=z,
                            name='solid', type=1,
                            bdry=bdry,)

    pa.constants['E'] = 1e9
    pa.constants['nu'] = 0.3
    pa.constants['G'] = pa.constants['E']/(2.0*(1+pa.constants['nu']))
    pa.constants['K'] = stress_funcs.get_K(pa.constants['G'], pa.constants['nu'])
    pa.constants['rho0'] = rho0
    pa.constants['dr0'] = dx
    pa.constants['c_s'] = numpy.sqrt(pa.constants['K']/pa.constants['rho0'])
    pa.cs = numpy.ones_like(x) * pa.constants['c_s']
    pa.set(idx=numpy.arange(len(pa.x)))
    print 'G:', pa.G
    print 'K', pa.K
    print 'c_s', pa.c_s
    print 'Number of particles: ', len(pa.x)

    return pa


class FixedBoundary(SPHFunction):
    def __init__(self, source, dest, particle_indices, props=['x','y','z'],
                 values=[0,0,0], setup_arrays=True):
        self.indices = particle_indices
        self.props = props
        self.values = values
        SPHFunction.__init__(self, source, dest, setup_arrays)
    
    def set_src_dst_reads(self):
        self.src_reads = self.dst_reads = self.props
        
    def eval(self, solver):
        for i,prop in enumerate(self.props):
            self.dest.get(prop)[self.indices] = self.values[i]

# use the solvers default cubic spline kernel
s = StressSolver(dim=1, integrator_type=solver.PredictorCorrectorIntegrator, xsph=0.5, marts_eps=0.3, marts_n=4, CFL=None)

# can be overriden by commandline arguments
s.set_time_step(1e-7)
s.set_final_time(1e-3)

app.setup(s, create_particles=create_particles)
particles = s.particles
pa = particles.arrays[0]

s.pre_step_functions.append(FixedBoundary(pa, pa, props=['u','x'], values=[0,0],
                                      particle_indices=numpy.flatnonzero(pa.bdry)))

for i in range(len(particles.arrays[0].x)):
    app.command_manager.add_function(PrintPos(i, ['x','y','u','p','rho','sigma00','ubar'],
                  s.output_directory+'/stress%s.dat'%i).function,
                                     interval=1)

s.set_kernel_correction(-1)
s.pfreq = 10

app.run()

sys.exit(0)

from pylab import *
pa = particles.arrays[0]

plot(pa.x, pa.y, '.', label='y')
legend(loc='best')

figure()
plot(pa.x, pa.u, '.', label='u')
legend(loc='best')

figure()
plot(pa.x, pa.ubar, '.', label='ubar')
legend(loc='best')

figure()
plot(pa.x, pa.rho, '.', label='rho')
legend(loc='best')

figure()
plot(pa.x, pa.p, '.', label='p')
legend(loc='best')

figure()
plot(pa.x, pa.sigma00, '.', label='sigma00')
legend(loc='best')


print pa.x
print pa.y
print pa.z
print pa.u
print pa.v
print pa.w

show()

