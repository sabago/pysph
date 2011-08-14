"""1D shock tube problem which simulates the collision of two strong
shocks. The test is described in 'An adaptive SPH method for strong
shocks' by Leonardo Di. G. Sigalotti and Henri Lopez and Leonardo
Trujillo, JCP, vol 228, pp (5888-5907)

"""

import pysph.solver.api as solver
import pysph.base.api as base
import pysph.sph.api as sph

import numpy

import get_shock_tube_data as get_data

# Parameters
xl = -1.5; xr = 1.5
pl = 460.894; pr = 46.0950
ul = 19.5975; ur = -6.19633
rhol = 5.999242; rhor = 5.999242

# Number of particles
nl = 500*3
nr = 500*3
np = nl + nr

# Time step constants
dt = 5e-6
tf = 0.035

# Artificial Viscosity constants
alpha = 1.0
beta = 1.0
gamma = 1.4
eta = 0.1

# ADKE Constants
eps = 0.5
k=1.0

D = 1.5
dx = 0.5/500
h0 = D*dx

# mass
m0 = rhol*dx

# Artificial Heat constants
g1 = 0.5
g2 = 0.5

def get_particles(with_boundary=True, **kwargs):
    
    adke, left, right = get_data.get_shock_tube_data(nl=nl,nr=nr,xl=xl, xr=xr,
                                                     pl=pl, pr=pr,
                                                     rhol=rhol, rhor=rhor,
                                                     ul=ul, ur=ur,
                                                     g1=g1, g2=g2, h0=h0,
                                                     gamma=1.4)

    adke.m[:] = m0
    
    return [adke,]

app = solver.Application()

s = solver.ADKEShockTubeSolver(dim=1,
                               integrator_type=solver.RK2Integrator,
                               h0=h0, eps=eps, k=k, g1=g1, g2=g2,
                               alpha=alpha, beta=beta)

s.set_final_time(tf)
s.set_time_step(dt)

app.setup(
    solver=s,
    min_cell_size=4*h0,
    variable_h=True,
    create_particles=get_particles)

output_dir = app.options.output_dir
numpy.savez(output_dir + "/parameters.npz", eps=eps, k=k, h0=h0,
            g1=g1, g2=g2, alpha=alpha, beta=beta)

app.run()
