""" Cylindrical Noh's implosion problem using the ADKE algorithm.

Particles are distributed on concentric circles about the origin with
increasing number of particles with increasing radius. The velocity is
initially uniform and directed towards the origin.

"""

import numpy
import pysph.sph.api as sp
import pysph.base.api as base
import pysph.solver.api as solver

pi = numpy.pi
cos = numpy.cos
sin = numpy.sin

gamma = 5.0/3.0
alpha = 1.0
beta = 1.0

k = 0.9
eps = 0.4

g1 = 0.5
g2 = 1.0

dt = 1e-4
tf = 0.6

n = 120
dr = 1.0/n

h0 = dr
rho0 = 1.0
m1 = pi*dr*dr*rho0/4

def create_particles(**kwargs):
    x = numpy.zeros(0)
    y = numpy.zeros(0)
    u = numpy.zeros(0)
    v = numpy.zeros(0)
    m = numpy.zeros(0)

    rad = 0.0

    for j in range(1, n+1):
        npnts = 4*j
        dtheta = 2*pi/npnts

        theta = numpy.arange(0, 2*pi-1e-10, dtheta)
        rad = rad + dr

        _x = rad*cos(theta)
        _y = rad*sin(theta)

        _u = -cos(theta)
        _v = -sin(theta)

        if j == 1:
            _m = numpy.ones_like(_x) * m1
        else:
            _m = numpy.ones_like(_x) * (2.0*j - 1.0)/(j) * m1

        x = numpy.concatenate( (x, _x) )
        y = numpy.concatenate( (y, _y) )
        m = numpy.concatenate( (m, _m) )
        u = numpy.concatenate( (u, _u) )
        v = numpy.concatenate( (v, _v) )

    rho = numpy.ones_like(x) * 1.0
    h = numpy.ones_like(x) * h0
    p = numpy.ones_like(x) * 0.0
    e = numpy.ones_like(x) * 0.0
    rhop = numpy.ones_like(x)
    div = numpy.zeros_like(x)
    q = numpy.zeros_like(x)

    fluid = base.get_particle_array(name="fluid", type=base.Fluid,
                                    x=x,y=y,m=m,rho=rho, h=h,
                                    u=u,v=v,p=p,e=e,
                                    rhop=rhop, q=q, div=div)

    print "Number of fluid particles = ", fluid.get_number_of_particles()

    return fluid

app = solver.Application()

s = solver.ADKEShockTubeSolver(dim=2,
                               integrator_type=solver.RK2Integrator,
                               h0=h0, eps=eps, k=k, g1=g1, g2=g2,
                               alpha=alpha, beta=beta, gamma=gamma)

s.set_final_time(tf)
s.set_time_step(dt)

app.setup(
    solver=s,
    min_cell_size=4*h0,
    variable_h=True,
    create_particles=create_particles)

output_dir = app.options.output_dir
numpy.savez(output_dir + "/parameters", eps=eps, k=k, h0=h0,
            g1=g1, g2=g2, alpha=alpha, beta=beta,
            gamma=gamma, hks=app.options.hks, kernel=app.options.kernel)
app.run()
