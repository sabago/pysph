""" An example solving the Ellptical drop test case """

import pysph.base.api as base
import pysph.solver.api as solver

import warnings

dt = 1e-4
tf = 0.0076

app = solver.Application()

# set the integrator type
integrator_type = solver.RK2Integrator

s = solver.FluidSolver(dim=2, integrator_type=integrator_type)

s.set_time_step(dt)
s.set_final_time(tf)

app.setup(
    solver=s,
    variable_h=False,
    create_particles=solver.fluid_solver.get_circular_patch, name='fluid', type=0,
    locator_type=base.NeighborLocatorType.SPHNeighborLocator,
    cl_locator_type=base.OpenCLNeighborLocatorType.LinkedListSPHNeighborLocator,
    domain_manager_type=base.DomainManagerType.LinkedListManager)

if app.options.with_cl:
    msg = """\n\n
You have chosen to run the example with OpenCL support.  The only
integrator with OpenCL support is the forward Euler
integrator. This integrator will be used instead of the default
RK2 integrator for this example.\n\n
"""
    warnings.warn(msg)
    integrator_type = solver.EulerIntegrator

# Print the output at every time step
s.set_print_freq(1)

app.run()



