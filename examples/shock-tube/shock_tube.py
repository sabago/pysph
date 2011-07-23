""" An example script for running the shock tube problem using Standard
SPH.

Global properties for the shock tube problem:
---------------------------------------------
x ~ [-.6,.6], dxl = 0.001875, dxr = dxl*4, m = dxl, h = 2*dxr
rhol = 1.0, rhor = 0.25, el = 2.5, er = 1.795, pl = 1.0, pr = 0.1795


These are obtained from the solver.shock_tube_solver.standard_shock_tube_data
"""
import logging

import pysph.base.api as base
import pysph.solver.api as solver
from pysph.base.kernels import CubicSplineKernel

CLDomain = base.DomainManagerType
CLLocator = base.OpenCLNeighborLocatorType
Locator = base.NeighborLocatorType

nl = 320
nr = 80

# Create the application, do this first so the application sets up the
# logging and also gets all command line arguments.
app = solver.Application()

# Set the solver using the default cubic spline kernel
s = solver.ShockTubeSolver(dim=1, integrator_type=solver.EulerIntegrator)
# set the default solver constants.
s.set_final_time(0.15)
s.set_time_step(3e-4)

# Set the application's solver.  We do this at the end since the user
# may have asked for a different timestep/final time on the command
# line.
app.setup(
    solver=s,
    variable_h=False,
    create_particles=solver.shock_tube_solver.standard_shock_tube_data,
    name='fluid', type=0,
    locator_type=Locator.SPHNeighborLocator,
    cl_locator_type=CLLocator.AllPairNeighborLocator,
    domain_manager_type=CLDomain.DomainManager,
    nl=nl, nr=nr, smoothing_length=None)

# Run the application.
app.run()

