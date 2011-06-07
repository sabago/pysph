""" Shock tube problem with the ADKE procedure of Sigalotti """

import pysph.solver.api as solver
import pysph.base.api as base
import pysph.sph.api as sph

from pysph.base.kernels import CubicSplineKernel
import numpy

Fluid = base.ParticleType.Fluid
Boundary = base.ParticleType.Boundary

# Shock tube parameters

dxl = 0.6/320
dxr = 4*dxl

h0 = 2*dxr
eps = 0.4
k = 0.7

g1 = 0.5
g2 = 0.5

class UpdateBoundaryParticles:
    def __init__(self, particles):
        self.particles = particles

    def eval(self):
        left = particles.get_named_particle_array('left')
        right = particles.get_named_particle_array("right")

        fluid = particles.get_named_particle_array("fluid")

        left.h[:] = fluid.h[0]
        right.h[:] = fluid.h[-1]

def get_fluid_particles(**kwargs):
    pa = solver.shock_tube_solver.standard_shock_tube_data(name="fluid")

    pa.add_property({'name':'rhop','type':'double'})
    pa.add_property({'name':'div', 'type':'double'})
    pa.add_property( {'name':'q', 'type':'double'} )

    return pa

def get_boundary_particles(**kwargs):

    # left boundary
    x = numpy.ones(50)
    for i in range(50):
        x[i] = -0.6 - (i+1) * dxl

    m = numpy.ones_like(x) * dxl
    h = numpy.ones_like(x) * 2*dxr
    rho = numpy.ones_like(x)
    u = numpy.zeros_like(x)
    e = numpy.ones_like(x) * 2.5

    p = (0.4) * rho * e
    cs = numpy.sqrt(1.4*p/rho)
    q = g1 * h * cs

    left = base.get_particle_array(name="left", type=Boundary,
                                   x=x, m=m, h=h, rho=rho, u=u,
                                   e=e, cs=cs, p=p, q=q)

    # right boundary
    for i in range(50):
        x[i] = 0.6 + (i + 1)*dxr

    m = numpy.ones_like(x) * dxl
    h = numpy.ones_like(x) * 2*dxr
    rho = numpy.ones_like(x) * 0.25
    u = numpy.zeros_like(x)
    e = numpy.ones_like(x) * 1.795

    p = (0.4) * rho * e
    cs = numpy.sqrt(1.4*p/rho)
    q = g1 * h * cs

    right = base.get_particle_array(name="right", type=Boundary,
                                    x=x, m=m, h=h, rho=rho, u=u,
                                    e=e, cs=cs,p=p, q=q)

    return [left, right]

def get_particles(**kwargs):
    particles = []
    particles.append(get_fluid_particles())

    particles.extend(get_boundary_particles())

    return particles

# Create the application

app = solver.Application()
app.process_command_line()

particles = app.create_particles(
    variable_h=True, callable=get_particles, 
    name='fluid', type=Fluid,
    locator_type=base.NeighborLocatorType.SPHNeighborLocator)

# add the boundary update function to the particles
particles.add_misc_function( UpdateBoundaryParticles(particles) )

# define the solver and kernel
s = solver.Solver(dim=1, integrator_type=solver.EulerIntegrator)

#############################################################
#                     ADD OPERATIONS
#############################################################

# pilot rho
s.add_operation(solver.SPHOperation(

    sph.ADKEPilotRho.withargs(h0=h0),
    on_types=[Fluid], from_types=[Fluid,Boundary],
    updates=['rhop'], id='adke_rho'),

                )

# smoothing length update
s.add_operation(solver.SPHOperation(

    sph.ADKESmoothingUpdate.withargs(h0=h0, k=k, eps=eps),
    on_types=[Fluid], updates=['h'], id='adke'),
                
                )

# summation density
s.add_operation(solver.SPHOperation(

    sph.SPHRho.withargs(),
    from_types=[Fluid, Boundary], on_types=[Fluid], 
    updates=['rho'], id = 'density')
                
                )

# ideal gas equation
s.add_operation(solver.SPHOperation(
    
    sph.IdealGasEquation.withargs(),
    on_types = [Fluid], updates=['p', 'cs'], id='eos')
                
                )

# velocity divergence
s.add_operation(solver.SPHOperation(

    sph.VelocityDivergence.withargs(),
    on_types=[Fluid], from_types=[Fluid, Boundary],
    updates=['div'], id='vdivergence'),

                )

# conduction coefficient update
s.add_operation(solver.SPHOperation(

    sph.ADKEConductionCoeffUpdate.withargs(g1=g1, g2=g2),
    on_types=[Fluid],
    updates=['q'], id='qcoeff'),

                )


# momentum equation
s.add_operation(solver.SPHIntegration(
    
    sph.MomentumEquation.withargs(),
    from_types=[Fluid, Boundary], on_types=[Fluid], 
    updates=['u'], id='mom')
                
                )

# energy equation
s.add_operation(solver.SPHIntegration(
    
    sph.EnergyEquation.withargs(hks=False),
    from_types=[Fluid, Boundary],
    on_types=[Fluid], updates=['e'], id='enr')

                )

# artificial heat 
s.add_operation(solver.SPHIntegration(

   sph.ArtificialHeat.withargs(eta=0.1),
   on_types=[Fluid], from_types=[Fluid,Boundary],
   updates=['e'], id='aheat'),

               )

# position step
s.add_operation_step([Fluid])

s.set_final_time(0.15)
s.set_time_step(3e-4)

app.set_solver(s)

app.run()
