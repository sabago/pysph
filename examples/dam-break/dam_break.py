""" 2D Dam Break Over a dry bed. The case is described in "State of
the art classical SPH for free surface flows", Benedict D Rogers,
Robert A, Dalrymple and Alex J.C Crespo, Journal of Hydraulic
Research, Vol 48, Extra Issue (2010), pp 6-27


Setup:
------



x                   x !
x                   x !
x                   x !
x                   x !
x  o   o   o        x !
x    o   o          x !3m
x  o   o   o        x !
x    o   o          x !
x  o   o   o        x !
x                   x !
xxxxxxxxxxxxxxxxxxxxx |        o -- Fluid Particles
                               x -- Solid Particles
     -dx-                      dx = dy
_________4m___________

Y
|
|
|
|
|
|      /Z
|     /
|    /
|   /
|  /
| /
|/_________________X

Fluid particles are placed on a staggered grid. The nodes of the grid
are located at R = l*dx i + m * dy j with a two point bias (0,0) and
(dx/2, dy/2) refered to the corner defined by R. l and m are integers
and i and j are the unit vectors alon `X` and `Y` respectively.

For the Monaghan Type Repulsive boundary condition, a single row of
boundary particles is used with a boundary spacing delp = dx = dy.

For the Dynamic Boundary Conditions, a staggered grid arrangement is
used for the boundary particles.

Numerical Parameters:
---------------------

dx = dy = 0.012m
h = 0.0156 => h/dx = 1.3

Height of Water column = 2m
Length of Water column = 1m

Number of particles = 27639 + 1669 = 29308


ro = 1000.0
co = 10*sqrt(2*9.81*2) ~ 65.0
gamma = 7.0

Artificial Viscosity:
alpha = 0.5

XSPH Correction:
eps = 0.5

 """

import warnings

import numpy

import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

from pysph.tools import geometry_utils as geom

Fluid = base.ParticleType.Fluid
Solid = base.ParticleType.Solid

fluid_column_height = 2.0
fluid_column_width  = 1.0
container_height = 3.0
container_width  = 4.0

#h = 0.0156
h = 0.0390
#h = 0.01
dx = dy = 0.03
ro = 1000.0
co = 10.0 * numpy.sqrt(2*9.81*fluid_column_height)
gamma = 7.0
alpha = 0.3
eps = 0.5

B = co*co*ro/gamma

def get_boundary_particles():
    """ Get the particles corresponding to the dam and fluids """
    
    xb1, yb1 = geom.create_2D_tank(x1=0, y1=0,
                                   x2=container_width, y2=container_height,
                                   dx=dx)
    
    xb2, yb2 = geom.create_2D_tank(x1=-dx/2, y1=-dx/2,
                                   x2=container_width, y2=container_height,
                                   dx=dx)

    xb = numpy.concatenate((xb1, xb2))
    yb = numpy.concatenate((yb1, yb2))

    hb = numpy.ones_like(xb)*h
    mb = numpy.ones_like(xb)*dx*dy*ro*0.5
    rhob = numpy.ones_like(xb) * ro

    cb = numpy.ones_like(xb)*co

    boundary = base.get_particle_array(cl_precision="single",
                                       name="boundary", type=Solid, 
                                       x=xb, y=yb, h=hb, rho=rhob, cs=cb,
                                       m=mb)

    print 'Number of Boundary particles: ', len(xb)

    return boundary

def get_fluid_particles():
    
    xf1, yf1 = geom.create_2D_filled_region(x1=dx, y1=dx,
                                            x2=fluid_column_width,
                                            y2=fluid_column_height,
                                            dx=dx)

    xf2, yf2 = geom.create_2D_filled_region(x1=dx/2, y1=dx/2,
                                            x2=fluid_column_width,
                                            y2=fluid_column_height,
                                            dx=dx)
    

    x = numpy.concatenate((xf1, xf2))
    y = numpy.concatenate((yf1, yf2))

    print 'Number of fluid particles: ', len(x)

    hf = numpy.ones_like(x) * h
    mf = numpy.ones_like(x) * dx * dy * ro * 0.5
    rhof = numpy.ones_like(x) * ro
    csf = numpy.ones_like(x) * co
    
    fluid = base.get_particle_array(cl_precision="single",
                                    name="fluid", type=Fluid,
                                    x=x, y=y, h=hf, m=mf, rho=rhof,
                                    cs=csf)

    return fluid

def get_particles(**args):
    fluid = get_fluid_particles()
    boundary = get_boundary_particles()

    return [fluid, boundary]


app = solver.Application()

integrator_type = solver.PredictorCorrectorIntegrator
s = solver.Solver(dim=2, integrator_type=integrator_type)

kernel = base.CubicSplineKernel(dim=2)

# define the artificial pressure term for the momentum equation
deltap = dx
n = 4

#Equation of state
s.add_operation(solver.SPHOperation(
        
    sph.TaitEquation.withargs(hks=False, co=co, ro=ro),
    on_types=[Fluid, Solid], 
    updates=['p', 'cs'],
    id='eos'),
                
                )

#Continuity equation
s.add_operation(solver.SPHIntegration(
        
    sph.SPHDensityRate.withargs(hks=False),
    on_types=[Fluid, Solid], from_types=[Fluid, Solid], 
    updates=['rho'], id='density')
                
                )

#momentum equation
s.add_operation(solver.SPHIntegration(
        
    sph.MomentumEquation.withargs(alpha=alpha, beta=0.0, hks=False,
                                  deltap=None, n=n),
    on_types=[Fluid], from_types=[Fluid, Solid],  
    updates=['u','v'], id='mom')
                    
                 )

#s.add_operation(solver.SPHIntegration(
    
#    sph.SPHPressureGradient.withargs(),
#    on_types=[Fluid], from_types=[Fluid,Solid],
#    updates=['u','v'], id='pgrad')

#                )

#s.add_operation(solver.SPHIntegration(

#    sph.MonaghanArtificialVsicosity.withargs(alpha=alpha, beta=0.0),
#    on_types=[Fluid], from_types=[Fluid,Solid],
#    updates=['u','v'], id='avisc')

#                )


#Gravity force
s.add_operation(solver.SPHIntegration(
        
    sph.GravityForce.withargs(gy=-9.81),
    on_types=[Fluid],
    updates=['u','v'],id='gravity')
                
                )

# Position stepping and XSPH correction operations

s.add_operation_step([Fluid])
s.add_operation_xsph(eps=eps)

dt = 1e-4

s.set_final_time(3.0)
s.set_time_step(dt)

app.setup(
    solver=s,
    variable_h=False, create_particles=get_particles, min_cell_size=4*h,
    locator_type=base.NeighborLocatorType.SPHNeighborLocator,
    domain_manager=base.DomainManagerType.DomainManager,
    cl_locator_type=base.OpenCLNeighborLocatorType.AllPairNeighborLocator
    )

# this tells the solver to compute the max time step dynamically
s.time_step_function = solver.ViscousTimeStep(co=co,cfl=0.3,
                                              particles=s.particles)

#s.time_step_function = solver.ViscousAndForceBasedTimeStep(co=co, cfl=0.3,
#                                                           particles=s.particles)

if app.options.with_cl:
    msg = """\n\n
You have chosen to run the example with OpenCL support.  The only
integrator with OpenCL support is the forward Euler
integrator. This integrator will be used instead of the default
predictor corrector integrator for this example.\n\n
"""
    warnings.warn(msg)
    integrator_type = solver.EulerIntegrator

app.run()
