""" An example solving stress test case """

import sys
import numpy
from numpy import pi, sin, sinh, cos, cosh

import pysph.base.api as base
import pysph.sph.api as sph
import pysph.solver.api as solver
from pysph.solver.stress_solver import StressSolver, get_particle_array
from pysph.sph.funcs import stress_funcs
from pysph.sph.api import SPHFunction

app = solver.Application()

#dt = app.options.time_step if app.options.time_step else 1e-8
CFL = 0.1
dim = 3
#tf = app.options.final_time if app.options.final_time else 1e-2

class PrintPos(object):
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
    #x,y = numpy.mgrid[-1.05:1.05+1e-4:dx, -0.105:0.105+1e-4:dx]
    dx = 0.002 # 2mm
    R = 0.02
    xl = -0.05
    L = 0.2

    x,y,z = numpy.mgrid[xl:L+dx/2:dx, -R/2:(R+dx)/2:dx, -R/2:(R+dx)/2:dx]
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    r2 = y**2+z**2
    keep = numpy.flatnonzero(r2<R*R/4)
    x = x[keep]
    y = y[keep]
    z = z[keep]

    bdry = (x<dx/2)*1.0
    bdry_indices = numpy.flatnonzero(bdry)
    rod_indices = numpy.flatnonzero(1-bdry)
    x2 = x[bdry_indices]
    y2 = y[bdry_indices]
    z2 = z[bdry_indices]
    x = x[rod_indices]
    y = y[rod_indices]
    z = z[rod_indices]

    print 'num_particles:', len(x), 'num_bdry_particles:', len(x2)
    #print bdry, numpy.flatnonzero(bdry)
    m = numpy.ones_like(x)*dx**dim
    m2 = numpy.ones_like(x2)*dx**dim
    h = numpy.ones_like(x)*1.5*dx
    h2 = numpy.ones_like(x2)*1.5*dx
    rho = numpy.ones_like(x)
    rho2 = numpy.ones_like(x2)

    p = u = x*0
    vel_max = 1
    v = z*vel_max/max(z)*sin(pi*x/2/L)
    w = -y*vel_max/max(y)*sin(pi*x/2/L)
    p2 = u2 = v2 = w2 = x2*0


    pa = get_particle_array(x=x, y=y, z=z, m=m, rho=rho, h=h, p=p, u=u, v=v, w=w,
                            name='solid',
                            )

    pa.constants['E'] = 1e9
    pa.constants['nu'] = 0.25
    pa.constants['G'] = pa.constants['E']/(2.0*(1+pa.constants['nu']))
    pa.constants['K'] = stress_funcs.get_K(pa.constants['G'], pa.constants['nu'])
    pa.constants['rho0'] = 1.0
    pa.constants['dr0'] = dx
    pa.constants['c_s'] = (pa.constants['K']/pa.constants['rho0'])**0.5
    pa.cs = numpy.ones_like(x) * pa.constants['c_s']
    print 'c_s:', pa.c_s
    print 'G:', pa.G/pa.c_s**2/pa.rho0

    print 'v_f:', pa.v[-1]/pa.c_s, '(%s)'%pa.v[-1]
    print 'T:', 2*numpy.pi/(pa.E*0.02**2*(1.875/0.2)**4/(12*pa.rho0*(1-pa.nu**2)))**0.5
    pa.set(idx=numpy.arange(len(pa.x)))
    print 'Number of particles: ', len(pa.x)
    #print 'CFL:', pa.c_s*dt/dx/2
    #print 'particle_motion:', -pa.u[-1]*dt

    # boundary particle array
    pb = get_particle_array(x=x2, x0=x2, y=y2, y0=y2, z=z2, z0=z2,
                            m=m2, rho=rho2,
                            h=h2, p=p2,
                            name='bdry', type=1,
                            )

    pb.constants['E'] = 1e7
    pb.constants['nu'] = 0.25
    pb.constants['G'] = pb.constants['E']/(2.0*(1+pb.constants['nu']))
    pb.constants['K'] = stress_funcs.get_K(pb.constants['G'], pb.constants['nu'])
    pb.constants['rho0'] = 1.0
    pb.constants['dr0'] = dx
    pb.constants['c_s'] = (pb.constants['K']/pb.constants['rho0'])**0.5
    pb.cs = numpy.ones_like(x2) * pb.constants['c_s']

    return [pa, pb]


class FixedBoundary(SPHFunction):
    def __init__(self, source, dest, props=['x','y','z'],
                 values=[0,0,0], setup_arrays=True):
        self.props = props[:]
        self.values = values[:]
        SPHFunction.__init__(self, source, dest, setup_arrays)
    
    def set_src_dst_reads(self):
        self.src_reads = self.dst_reads = self.props + [i for i in self.values if isinstance(i,str)]

    def eval(self, solver):
        for i,prop in enumerate(self.props):
            p = self.dest.get_carray(prop)
            p = p.get_npy_array()
            v = self.values[i]
            if isinstance(v, str):
                p[:] = getattr(self.dest, v)
            else:
                p[:] = v


# use the solvers default cubic spline kernel
# s = StressSolver(dim=2, integrator_type=solver.RK2Integrator)
# FIXME: LeapFrog Integrator does not work
s = StressSolver(dim=3, integrator_type=solver.EulerIntegrator, xsph=0.5,
                 marts_eps=0.3, marts_n=4, CFL=CFL)


# can be overriden by commandline arguments
dt = 1e-8
tf = 1e-2
s.set_time_step(dt)
s.set_final_time(tf)
s.set_kernel_correction(-1)
s.pfreq = 100

app.setup(s, create_particles=create_particles)

particles = s.particles
pa, pb = particles.arrays

s.pre_step_functions.append(FixedBoundary(pb, pb, props=['x','y','z','u','v','w','rho'],
                                          values=['x0','y0','z0',0,0,0,'rho0']))


app.run()

