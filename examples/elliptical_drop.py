""" An example solving the Ellptical drop test case """

import pysph.base.api as base
import pysph.solver.api as solver

dt = 1e-4
tf = 0.00076

app = solver.Application()

particles = app.create_particles(
    False,
    solver.fluid_solver.get_circular_patch, name='fluid', type=0,
    locator_type=base.NeighborLocatorType.SPHNeighborLocator,
    cl_locator_type=base.OpenCLNeighborLocatorType.LinkedListSPHNeighborLocator,
    domain_manager_type=base.DomainManagerType.LinkedListManager)

# use the solvers default cubic spline kernel
s = solver.FluidSolver(dim=2, integrator_type=solver.RK2Integrator)

s.set_time_step(dt)
s.set_final_time(tf)

app.set_solver(s)

# Print the output at every time step
s.set_print_freq(1)

app.run()



