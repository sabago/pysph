""" An example solving stress test case """

import numpy
import sys

import pysph.base.api as base
import pysph.solver.api as solver
from pysph.solver.stress_solver import StressSolver, get_particle_array
from pysph.sph.funcs import stress_funcs, arithmetic_funcs
from pysph.sph.api import SPHFunction

app = solver.Application()

class PrintPos(object):
    ''' print properties of a particle in a column format (gnuplot/np.loadtxt) '''
    def __init__(self, particle_id, props=['x'], filename='stress.dat', write_interval=100):
        self.file = open(filename, 'w')
        self.file.write('i\t'+'\t'.join(props)+'\n')
        self.res = []
        self.props = props
        self.particle_id = particle_id
        self.write_interval = write_interval

    def function(self, solver):
        l = [solver.count]
        for prop in self.props:
            l.append(getattr(solver.particles.arrays[0], prop)[self.particle_id])
        self.res.append(l)
        if solver.count%self.write_interval == 0:
            s = '\n'.join('\t'.join(map(str,line)) for line in self.res)
            self.file.write(s)
            self.file.write('\n')
            self.res = []

    __call__ = function


def create_particles():
    N = 21
    #x,y = numpy.mgrid[-1.05:1.05+1e-4:dx, -0.105:0.105+1e-4:dx]
    x = numpy.mgrid[0:1:1j*N]
    dx = 1.0/(N-1)
    x = x.ravel()
    #y = y.ravel()
    bdry = (x<=0)
    print bdry, numpy.flatnonzero(bdry)
    m = numpy.ones_like(x)*dx
    h = numpy.ones_like(x)*1.2*dx
    rho = numpy.ones_like(x)
    y = z = 1-rho

    p = 0.5*1.0*100*100*(1 - (x**2 + y**2))

    #cs = numpy.ones_like(x) * 10000.0

    u = -x
    u *= 0.01
    #u = numpy.ones_like(x)*
    p *= 0.0
    h *= 1
    v = 0.0*y

    pa = get_particle_array(x=x, y=y, m=m, rho=rho, h=h, p=p, u=u, v=v, z=z,w=z,
                            name='solid', type=1,
                            bdry=bdry,)

    pa.constants['E'] = 1e7
    pa.constants['nu'] = 0.3
    pa.constants['G'] = pa.constants['E']/(2.0*1+pa.constants['nu'])
    pa.constants['K'] = stress_funcs.get_K(pa.constants['G'], pa.constants['nu'])
    pa.constants['rho0'] = 1.
    pa.constants['dr0'] = dx
    pa.constants['c_s'] = numpy.sqrt(pa.constants['K']/pa.constants['rho0'])
    pa.cs = numpy.ones_like(x) * pa.constants['c_s']
    pa.set(idx=numpy.arange(len(pa.x)))
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
s = StressSolver(dim=1, integrator_type=solver.LeapFrogIntegrator)

# can be overriden by commandline arguments
s.set_time_step(1e-6)
s.set_final_time(1e-1)

app.set_solver(s, create_particles)
particles = s.particles
pa = particles.arrays[0]

s.pre_step_functions.append(FixedBoundary(pa, pa, props=['u','x'], values=[0,0],
                                      particle_indices=numpy.flatnonzero(pa.bdry)))

for i in range(len(particles.arrays[0].x)):
    app.command_manager.add_function(PrintPos(i, ['x','y','u','p','rho','sigma00','ubar'],
                                              'stress1d/stress%s.dat'%i, 100),
                                      interval=1)

s.set_kernel_correction(0)
s.add_print_properties(['sigma00'])
s.pfreq = 100

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

