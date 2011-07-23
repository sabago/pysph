""" An example solving stress test case """

import numpy
import sys

import pysph.base.api as base
import pysph.solver.api as solver
from pysph.solver.stress_solver import StressSolver, get_particle_array
from pysph.sph.funcs import stress_funcs, arithmetic_funcs
from pysph.sph.api import SPHFunction

app = solver.Application()

#dt = app.options.time_step if app.options.time_step else 1e-8
#tf = app.options.final_time if app.options.final_time else 1e-2

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


def create_particles():
    #N = 21
    dx = 0.1

    #x,y = numpy.mgrid[-1.05:1.05+1e-4:dx, -0.105:0.105+1e-4:dx]
    x,y = numpy.mgrid[-0.2:5.01:dx, -0.2:0.21:dx]
    x = x.ravel()
    y = y.ravel()
    bdry = (x<0.01)*1.0
    print 'num_particles', len(x)
    print bdry, numpy.flatnonzero(bdry)
    m = numpy.ones_like(x)*dx*dx
    h = numpy.ones_like(x)*1.4*dx
    rho = numpy.ones_like(x)
    z = numpy.zeros_like(x)

    p = 0.5*1.0*100*100*(1 - (x**2 + y**2))

    cs = numpy.ones_like(x) * 10000.0

    u = -x
    u *= 1e0
    h *= 1
    v = 0.0*y
    p *= 0.0

    pa = get_particle_array(x=x, y=y, m=m, rho=rho, h=h, p=p, u=u, v=v, z=z,w=z,
                                 bdry=bdry)
    pa.constants['E'] = 1e9
    pa.constants['nu'] = 0.3
    pa.constants['G'] = pa.constants['E']/(2.0*(1+pa.constants['nu']))
    pa.constants['K'] = stress_funcs.get_K(pa.constants['G'], pa.constants['nu'])
    pa.constants['rho0'] = 1.
    pa.constants['dr0'] = dx
    pa.constants['c_s'] = numpy.sqrt(pa.constants['K']/pa.constants['rho0'])
    pa.cs = numpy.ones_like(x) * pa.constants['c_s']
    pa.set(idx=numpy.arange(len(pa.x)))
    print 'G_mu', pa.G/pa.K
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
            p = self.dest.get(prop)
            p[self.indices] = self.values[i]


CFL=None
# use the solvers default cubic spline kernel
s = StressSolver(dim=2, integrator_type=solver.PredictorCorrectorIntegrator,
                 xsph=0.5, marts_eps=0.3, marts_n=4, CFL=CFL)

dt = 1e-8
tf = 1e-3
s.set_time_step(dt)
s.set_final_time(tf)
s.pfreq = 100

app.setup(s, create_particles=create_particles)

particles = s.particles
pa = particles.arrays[0]

s.pre_step_functions.append(FixedBoundary(pa, pa, props=['u'], values=[0],
                                      particle_indices=numpy.flatnonzero(pa.bdry)))

s.pre_step_functions.append(FixedBoundary(pa, pa, props=['v'], values=[0],
                                      particle_indices=range(len(pa.x))))

s.set_kernel_correction(-1)

app.run()

