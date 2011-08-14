""" Standard shock tube problem by Monaghan """

import numpy

import pysph.base.api as base
import pysph.solver.api as solver

import get_shock_tube_data as data

CLDomain = base.DomainManagerType
CLLocator = base.OpenCLNeighborLocatorType
Locator = base.NeighborLocatorType

kernel = base.CubicSplineKernel
hks=False

# shock tube parameters
xl = -1.0; xr = 1.0
pl = 1000; pr = 0.01
ul = 0.0; ur = 0.0
rhol = 1.0; rhor = 1.0

# Number of particles
nl = 1000
nr = 1000
np = nl + nr

# Time step constants
dt = 5e-6
tf = 0.0075
t = 0.0

# Artificial Viscosity constants
alpha = 1.0
beta = 1.0
gamma = 1.4
eta = 0.1

# ADKE Constants
eps = 0.5
k=1.0
h0 = 1.5*xr/nr

# Artificial Heat constants
g1 = 0.2
g2 = 0.4

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
                               alpha=alpha, beta=beta,
                               kernel=kernel, hks=hks)

s.set_final_time(tf)
s.set_time_step(dt)

app.setup(
    solver=s,
    min_cell_size=4*h0,
    variable_h=True,
    create_particles=get_particles,
    locator_type=Locator.SPHNeighborLocator,
    cl_locator_type=CLLocator.AllPairNeighborLocator,
    domain_manager_type=CLDomain.DomainManager,
    nl=nl, nr=nr)

output_dir = app.options.output_dir
numpy.savez(output_dir + "/parameters.npz", eps=eps, k=k, h0=h0,
            g1=g1, g2=g2, alpha=alpha, beta=beta, hks=hks)
app.run()
