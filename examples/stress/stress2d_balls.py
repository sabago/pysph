""" An example solving stress test case : colliding rubber balls """

import sys
import numpy
from numpy import pi, sin, sinh, cos, cosh

import pysph.base.api as base
import pysph.sph.api as sph
import pysph.solver.api as solver
from pysph.solver.stress_solver import StressSolver
from pysph.sph.funcs import stress_funcs
from pysph.sph.api import SPHFunction

app = solver.Application()

#dt = app.options.time_step if app.options.time_step else 1e-8
#tf = app.options.final_time if app.options.final_time else 1e-2

def create_particles(two_arr=False):
    #x,y = numpy.mgrid[-1.05:1.05+1e-4:dx, -0.105:0.105+1e-4:dx]
    dx = 0.001 # 1mm
    ri = 0.03 # 3cm inner radius
    ro = 0.04 # 4cm outer radius
    spacing = 0.041 # spacing = 2*5cm
    
    x,y = numpy.mgrid[-ro:ro:dx, -ro:ro:dx]
    x = x.ravel()
    y = y.ravel()
    
    d = (x*x+y*y)
    keep = numpy.flatnonzero((ri*ri<=d) * (d<ro*ro))
    x = x[keep]
    y = y[keep]

    print 'num_particles', len(x)*2
    
    if not two_arr:
        x = numpy.concatenate([x-spacing,x+spacing])
        y = numpy.concatenate([y,y])

    #print bdry, numpy.flatnonzero(bdry)
    m = numpy.ones_like(x)*dx*dx
    h = numpy.ones_like(x)*1.4*dx
    rho = numpy.ones_like(x)
    z = numpy.zeros_like(x)

    p = 0.5*1.0*100*100*(1 - (x**2 + y**2))

    cs = numpy.ones_like(x) * 10000.0

    # u is set later
    v = z
    u_f = 0.059

    p *= 0
    h *= 1
    #u = 0.1*numpy.sin(x*pi/2.0/5.0)
    #u[numpy.flatnonzero(x<0.01)] = 0
    pa = base.get_particle_array(x=x+spacing, y=y, m=m, rho=rho, h=h, p=p, u=z, v=v, z=z,w=z,
                                 ubar=z, vbar=z, wbar=z,
                                 name='right_ball', type=1,
                                 sigma00=z, sigma11=z, sigma22=z,
                                 sigma01=z, sigma12=z, sigma02=z,
                                 MArtStress00=z, MArtStress11=z, MArtStress22=z,
                                 MArtStress01=z, MArtStress12=z, MArtStress02=z,
                                 #bdry=bdry
                                 )

    pa.constants['E'] = 1e7
    pa.constants['nu'] = 0.3975
    pa.constants['G'] = pa.constants['E']/(2.0*(1+pa.constants['nu']))
    pa.constants['K'] = stress_funcs.get_K(pa.constants['G'], pa.constants['nu'])
    pa.constants['rho0'] = 1.0
    pa.constants['dr0'] = dx
    pa.constants['c_s'] = (pa.constants['K']/pa.constants['rho0'])**0.5
    pa.cs = numpy.ones_like(x) * pa.constants['c_s']
    print 'c_s:', pa.c_s
    print 'G:', pa.G/pa.c_s**2/pa.rho0
    pa.u = pa.c_s*u_f*(2*(x<0)-1)
    print 'u_f:', pa.u[0]/pa.c_s, '(%s)'%pa.u[0]

    pa.set(idx=numpy.arange(len(pa.x)))
    print 'Number of particles: ', len(pa.x)
    print 'CFL:', pa.c_s*dt/dx/2
    print 'particle_motion:', abs(pa.u[0]*dt)


    if two_arr:
        pb = base.get_particle_array(x=x-spacing, y=y, m=m, rho=rho, h=h, p=p, u=u, v=v, z=z,w=z,
                                     ubar=z, vbar=z, wbar=z,
                                     name='left_ball', type=1,
                                     sigma00=z, sigma11=z, sigma22=z,
                                     sigma01=z, sigma12=z, sigma02=z,
                                     MArtStress00=z, MArtStress11=z, MArtStress22=z,
                                     MArtStress01=z, MArtStress12=z, MArtStress02=z,
                                     #bdry=bdry
                                     )

        pb.constants['E'] = 1e7
        pb.constants['nu'] = 0.3975
        pb.constants['G'] = pb.constants['E']/(2.0*1+pb.constants['nu'])
        pb.constants['K'] = stress_funcs.get_K(pb.constants['G'], pb.constants['nu'])
        pb.constants['rho0'] = 1.0
        pb.constants['c_s'] = (pb.constants['K']/pb.constants['rho0'])**0.5
        pb.cs = numpy.ones_like(x) * pb.constants['c_s']
        print 'c_s:', pb.c_s
        print 'G:', pb.G/pb.c_s**2/pb.rho0
        print 'G_mu', pa.G/pa.K
        pa.u = pa.c_s*u_f*(2*(x<0)-1)
        print 'u_f:', pb.u[-1]/pb.c_s, '(%s)'%pb.u[-1]
        
        pb.set(idx=numpy.arange(len(pb.x)))
        print 'Number of particles: ', len(pb.x)

        return [pa, pb]
    else:
        return pa



cfl = 0.1
# use the solvers default cubic spline kernel
# s = StressSolver(dim=2, integrator_type=solver.RK2Integrator)
s = StressSolver(dim=2, integrator_type=solver.PredictorCorrectorIntegrator,
                 xsph=0.5, marts_eps=0.3, marts_n=4, CFL=cfl)


# can be overriden by commandline arguments
dt = 1e-8
tf = 1e-2
s.set_time_step(dt)
s.set_final_time(tf)
s.set_kernel_correction(-1)
s.pfreq = 100

app.setup(s, create_particles=create_particles)

particles = s.particles
pa = particles.arrays[0]

app.run()
