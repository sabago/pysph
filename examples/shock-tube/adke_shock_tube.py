""" Shock tube problem with the ADKE procedure of Sigalotti """

import pysph.solver.api as solver
import pysph.base.api as base
import pysph.sph.api as sph

from pysph.base.kernels import CubicSplineKernel
import numpy

Fluid = base.ParticleType.Fluid
Boundary = base.ParticleType.Boundary

# Shock tube parameters

nl = int(320 * 7.5)
nr = int(80 * 7.5)

dxl = 0.6/nl
dxr = 4*dxl

h0 = 2*dxr
eps = 0.4
k = 0.7

g1 = 0.2
g2 = 0.5

alpha = 1.0
beta = 1.0

hks = False

class UpdateBoundaryParticles:
    def __init__(self, particles):
        self.particles = particles

    def eval(self):
        left = self.particles.get_named_particle_array('left')
        right = self.particles.get_named_particle_array("right")

        fluid = self.particles.get_named_particle_array("fluid")

        left.h[:] = fluid.h[0]
        right.h[:] = fluid.h[-1]

def get_fluid_particles(**kwargs):
    pa = solver.shock_tube_solver.standard_shock_tube_data(
        name="fluid", nl=nl, nr=nr)

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
    #cs = numpy.sqrt(0.4 * e)
    cs = numpy.sqrt( 1.4*p/rho )
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
    cs = numpy.sqrt( 1.4*p/rho )
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


# define the solver and kernel
s = solver.Solver(dim=1, integrator_type=solver.RK2Integrator)

#############################################################
#                     ADD OPERATIONS
#############################################################

# set the smoothing length
s.add_operation(solver.SPHOperation(

    sph.SetSmoothingLength.withargs(h0=h0),
    on_types=[base.Fluid,],
    updates=["h"],
    id="setsmoothing")
                   
                   )

# pilot rho
s.add_operation(solver.SPHOperation(

    sph.ADKEPilotRho.withargs(h0=h0),
    on_types=[Fluid], from_types=[Fluid,Boundary],
    updates=['rhop'], id='adke_rho'),

                )

# smoothing length update
s.add_operation(solver.SPHOperation(

    sph.ADKESmoothingUpdate.withargs(h0=h0, k=k, eps=eps, hks=hks),
    on_types=[Fluid], updates=['h'], id='adke'),
                
                )

# summation density
s.add_operation(solver.SPHOperation(

    sph.SPHRho.withargs(hks=hks),
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

    sph.VelocityDivergence.withargs(hks=hks),
    on_types=[Fluid], from_types=[Fluid, Boundary],
    updates=['div'], id='vdivergence'),

                )

#conduction coefficient update
s.add_operation(solver.SPHOperation(

    sph.ADKEConductionCoeffUpdate.withargs(g1=g1, g2=g2),
    on_types=[Fluid],
    updates=['q'], id='qcoeff'),

                )

# momentum equation
s.add_operation(solver.SPHIntegration(
    
    sph.MomentumEquation.withargs(alpha=1, beta=1, hks=hks),
    from_types=[Fluid, Boundary], on_types=[Fluid], 
    updates=['u'], id='mom')
                
                )

# energy equation
s.add_operation(solver.SPHIntegration(
    
    sph.EnergyEquation.withargs(),
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

app.setup(
    solver=s,
    min_cell_size = 4*h0,
    variable_h=True, create_particles=get_particles,
    locator_type=base.NeighborLocatorType.SPHNeighborLocator
    )

# add the boundary update function to the particles
s.particles.add_misc_function( UpdateBoundaryParticles(s.particles) )

output_dir = app.options.output_dir
numpy.savez(output_dir + "/parameters.npz", eps=eps, k=k, h0=h0,
            g1=g1, g2=g2, alpha=alpha, beta=beta, hks=hks)

app.run()
