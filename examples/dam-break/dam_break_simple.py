""" A tiny dam break problem


Setup:
------



x                   x !
x                   x !
x                   x !
x                   x !
x  o o o o o        x !
x  o o o o o        x !3m
x  o o o o o        x !
x  o o o o o        x !
x  o o o o o        x !
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


The Monaghan Type Repulsive boundary condition, with a single row of
boundary particles is used with a boundary spacing delp = dx = dy.

Numerical Parameters:
---------------------

h = 0.05
dx = dy = h/1.25 = 0.04

Height of Water column = 2m
Length of Water column = 1m

Number of fluid particles = 1250

ro = 1000.0
co = 10*sqrt(2*9.81*2) ~ 65.0
gamma = 7.0

Artificial Viscosity:
alpha = 0.5

XSPH Correction:
eps = 0.5

 """
import sys
import numpy
import pysph.base.api as base
import pysph.solver.api as solver
import pysph.sph.api as sph

Fluid = base.ParticleType.Fluid
Solid = base.ParticleType.Solid

h = 0.05
dx = dy = h/1.25
ro = 1000.0
co = 65.0
gamma = 7.0
alpha = 0.5
eps = 0.5

fluid_column_height = 2.0
fluid_column_width  = 1.0
container_height = 3.0
container_width  = 4.0

B = co*co*ro/gamma

def get_boundary_particles():
    """ Get the particles corresponding to the dam and fluids """
    
    left = base.Line(base.Point(0,0), container_height, numpy.pi/2)

    bottom = base.Line(base.Point(container_width,0), 
                       container_width, numpy.pi)
    
    right = base.Line(base.Point(container_width,container_height), 
                      container_height, 1.5*numpy.pi)

    g = base.Geometry('box', [left, bottom, right], is_closed=False)
    g.mesh_geometry(dx)
    boundary = g.get_particle_array(re_orient=False, 
                                    name="boundary")

    return boundary
    
def get_fluid_particles():
    
    xarr = numpy.arange(dx, 1.0 + dx, dx)
    yarr = numpy.arange(dx, 2.0 + dx, dx)
    
    x,y = numpy.meshgrid( xarr, yarr )
    x, y = x.ravel(), y.ravel()                    

    print 'Number of fluid particles: ', len(x)

    hf = numpy.ones_like(x) * h
    mf = numpy.ones_like(x) * dx * dy * ro 
    rhof = numpy.ones_like(x) * ro
    csf = numpy.ones_like(x) * co
    
    fluid = base.get_particle_array(name="fluid", type=Fluid,
                                    x=x, y=y, h=hf, m=mf, rho=rhof, cs=csf)

    return fluid

def get_particles(**args):
    fluid = get_fluid_particles()
    boundary = get_boundary_particles()

    return [fluid, boundary]

app = solver.Application()
s = solver.Solver(dim=2, integrator_type=solver.EulerIntegrator)

#Equation of state
s.add_operation(solver.SPHOperation(
        
    sph.TaitEquation.withargs(hks=False, co=co, ro=ro),
    on_types=[Fluid], 
    updates=['p', 'cs'],
    id='eos'),
                
                )

#Continuity equation
s.add_operation(solver.SPHIntegration(
        
    sph.SPHDensityRate.withargs(hks=False),
    on_types=[Fluid], from_types=[Fluid], 
    updates=['rho'], id='density')
                
                )

#momentum equation
s.add_operation(solver.SPHIntegration(
        
    sph.MomentumEquation.withargs(alpha=alpha, beta=0.0, hks=False),
    on_types=[Fluid], from_types=[Fluid],  
    updates=['u','v'], id='mom')
                    
                 )

#Gravity force
s.add_operation(solver.SPHIntegration(
        
    sph.GravityForce.withargs(gy=-9.81),
    on_types=[Fluid],
    updates=['u','v'],id='gravity')
                
                )

#the boundary force
s.add_operation(solver.SPHIntegration(
        
        sph.MonaghanBoundaryForce.withargs(delp=dx),
        on_types=[Fluid], from_types=[Solid], updates=['u','v'],
        id='bforce')
                
                )

# Position stepping and XSPH correction operations

s.add_operation_step([Fluid])
s.add_operation_xsph(eps=eps)

dt = 1e-4

s.set_final_time(3.0)
s.set_time_step(dt)

app.setup(
    solver=s,
    variable_h=False, create_particles=get_particles, min_cell_size=2*h,
    locator_type=base.NeighborLocatorType.SPHNeighborLocator,
    domain_manager=base.DomainManagerType.DomainManager,
    cl_locator_type=base.OpenCLNeighborLocatorType.AllPairNeighborLocator
    )

if app.options.with_cl:
    raise RuntimeError("OpenCL support not added for MonaghanBoundaryForce!")
                    
s.set_print_freq(1000)

app.run()
