""" Strong blaswave problem proposed by Sigalotti. Mach number = 771 """

import numpy

import pysph.base.api as base
import pysph.solver.api as solver

import get_shock_tube_data as data

Locator = base.NeighborLocatorType

kernel = base.CubicSplineKernel
hks=False

# shock tube parameters
xl = -1.5; xr = 1.5
pl = 1e4; pr = 0.01
ul = 0.0; ur = 0.0
rhol = 1.0; rhor = 1.0

# Number of particles
nl = 1500
nr = 1500
np = nl + nr

# Time step constants
dt = 5e-6
tf = 4e-3
t = 0.0

# Artificial Viscosity constants
alpha = 1.0
beta = 1.0
gamma = 5.0/3.0
eta = 0.1

# ADKE Constants
eps = 0.8
k=1.0
dx = xr/nr
D = 1.5
h0 = D*dx

# Artificial Heat constants
g1 = 0.2
g2 = 1.0

def get_particles(with_boundary=False, **kwargs):
    adke, left, right = data.get_shock_tube_data(nl=nl, nr=nr, xl=xl, xr=xr,
                                                 pl=pl, pr=pr,
                                                 rhol=rhol, rhor=rhor,
                                                 ul=ul, ur=ur,
                                                 g1=g1, g2=g2, h0=h0,
                                                 gamma=gamma)

    if with_boundary:
        return [adke, left, right]
    else:
       return [adke,]

app = solver.Application()

s = solver.ADKEShockTubeSolver(dim=1,
                               integrator_type=solver.RK2Integrator,
                               h0=h0, eps=eps, k=k, g1=g1, g2=g2,
                               alpha=alpha, beta=beta,gamma=gamma,
                               kernel=kernel, hks=hks,)

s.set_final_time(tf)
s.set_time_step(dt)

app.setup(
    solver=s,
    min_cell_size=4*h0,
    variable_h=True,
    create_particles=get_particles,
    locator_type=Locator.SPHNeighborLocator)

output_dir = app.options.output_dir
numpy.savez(output_dir + "/parameters.npz", eps=eps, k=k, h0=h0,
            g1=g1, g2=g2, alpha=alpha, beta=beta, hks=hks)
app.run()
