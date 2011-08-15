"""Woodward and COllela interacting blast wave."""

import numpy

import pysph.sph.api as sph
import pysph.base.api as base
import pysph.solver.api as solver

xl = 0
xr = 1.0
np = 5001

nbp = 100

dx = (xr-xl)/(np-1)
D = 1.5
h0 = D*dx

adke_eps = 0.5
adke_k = 1.0
g1 = 0.2
g2 = 0.4

alpha = 1.0
beta = 1.0
gamma = 1.4

tf = 0.04
dt = 2.5e-6

class UpdateBoundaryParticles(object):

    def __init__(self, particles, dx):
        self.particles = particles
        self.dx = dx

    def eval(self):
        left = self.particles.get_named_particle_array("left")
        right = self.particles.get_named_particle_array("right")
        fluid = self.particles.get_named_particle_array("fluid")

        left.h[:nbp] = fluid.h[:nbp]
        right.h[-nbp:] = fluid.h[-nbp:]

        left.u[:nbp] = -fluid.u[:nbp]
        right.u[-nbp:] = -fluid.u[-nbp:]

        left.e[:nbp] = fluid.e[:nbp]
        right.e[-nbp:] = fluid.e[-nbp:]

        left.p[:nbp] = fluid.p[:nbp]
        right.p[-nbp:] = fluid.p[-nbp:]

        left.rho[:nbp] = fluid.rho[:nbp]
        right.rho[-nbp:] = fluid.rho[-nbp:]

        left.cs[:nbp] = fluid.cs[:nbp]
        right.cs[-nbp:] = fluid.cs[-nbp:]

        left.q[:nbp] = fluid.q[:nbp]
        right.q[-nbp:] = fluid.q[-nbp:]
        
def get_particles(**kwargs):

    xleft = numpy.arange(xl, 0.1-dx+1e-10, dx)
    pleft = numpy.ones_like(xleft) * 1000.0

    xmid = numpy.arange(0.1+dx, 0.9-dx+1e-10, dx)
    pmid = numpy.ones_like(xmid) * 0.01

    xright = numpy.arange(0.9+dx, 1.0+1e-10, dx)
    pright = numpy.ones_like(xright) * 100.0

    x = numpy.concatenate( (xleft, xmid, xright) )
    p = numpy.concatenate( (pleft, pmid, pright) )
    rho = numpy.ones_like(x)

    m = numpy.ones_like(x) * dx
    h = numpy.ones_like(x) * D * dx

    e = p/( rho*(gamma-1.0) )
    cs = numpy.sqrt(gamma*p/rho)

    u = numpy.zeros_like(x)

    rhop = numpy.ones_like(x)
    div = numpy.zeros_like(x)
    q = g1 * h * cs

    fluid = base.get_particle_array(name="fluid", type=base.Fluid,
                                    x=x, m=m, h=h, rho=rho,
                                    p=p, e=e, cs=cs, u=u,
                                    rhop=rhop, div=div, q=q)

    nbp = 100
    x = numpy.ones(nbp)

    for i in range(nbp):
        x[i] = xl - (i+1)*dx

    m = numpy.ones_like(x) * fluid.m[0]
    p = numpy.ones_like(x) * fluid.p[0]
    rho = numpy.ones_like(x) * fluid.rho[0]
    h = numpy.ones_like(x) * fluid.p[0]

    e = p/( (gamma-1.0)*rho )
    cs = numpy.sqrt(gamma*p/rho)

    div = numpy.zeros_like(x)
    q = g1 * h * cs

    left = base.get_particle_array(name="left", type=base.Boundary,
                                   x=x, p=p, rho=rho, m=m, h=h,
                                   e=e, cs=cs, div=div, q=q)

    x = numpy.ones(nbp)
    _xr = xr + (nbp+1)*dx
    for i in range(nbp):
        x[i] = _xr - i*dx

    m = numpy.ones_like(x) * fluid.m[-1]
    p = numpy.ones_like(x) * fluid.p[-1]
    h = numpy.ones_like(x) * fluid.h[-1]
    rho = numpy.ones_like(x) * fluid.rho[-1]

    e = p/( (gamma-1.0)*rho )
    cs = numpy.sqrt(gamma*p/rho)

    div = numpy.zeros_like(x)
    q = g1 * h * cs

    right = base.get_particle_array(name="right", type=base.Boundary,
                                   x=x, p=p, rho=rho, m=m, h=h,
                                   e=e, cs=cs, div=div, q=q)    

    return [fluid,left,right]

app = solver.Application()

s = solver.ADKEShockTubeSolver(dim=1,
                               integrator_type=solver.RK2Integrator,
                               h0=h0, eps=adke_eps, k=adke_k, g1=g1, g2=g2,
                               alpha=alpha, beta=beta,gamma=gamma)

s.set_final_time(tf)
s.set_time_step(dt)

app.setup(
    solver=s,
    min_cell_size=6*h0,
    variable_h=True,
    create_particles=get_particles)

# add the boundary update function
s.particles.add_misc_function( UpdateBoundaryParticles(s.particles, dx) )

app.run()
