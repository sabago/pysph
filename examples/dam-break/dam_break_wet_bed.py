""" Dam break simulation over a wet bed.

This is part of the SPHERIC validation test cases (case 5)
(http://wiki.manchester.ac.uk/spheric/index.php/SPHERIC_Home_Page)

The main reference for this test case is 'State-of-the-art classical SPH for free-surface flows' by Moncho Gomez-Gesteira and Benedict D. Rogers and Robert
 A. Dalrymple and Alex J. Crespo, Journal of Hydraulic Research Extra
 Issue (2010) pp 6-27

"""

import numpy

import pysph.solver.api as solver
import pysph.base.api as base
import pysph.sph.api as sph

import pysph.tools.geometry_utils as geom

# Geometric parameters
dx = 0.005
h0 = 0.006
d = 0.0180
H = 0.15

tank_length = 0.38 + 3.0 #9.55
tank_height = 0.2

# Numerical parameters
vmax = numpy.sqrt(2*9.81*H)
co = 10.0 * vmax
ro = 1000.0
B = co*co*ro/7.0

alpha = 0.08
beta = 0.0
eps = 0.5

Fluid = base.ParticleType.Fluid
Solid = base.ParticleType.Solid

def get_boundary_particles():
    """ Get the particles corresponding to the dam and fluids """

    # get the tank
    xt1, yt1 = geom.create_2D_tank(x1=0, y1=0,
                                   x2=tank_length, y2=tank_height,
                                   dx=dx)

    xt2,  yt2 = geom.create_2D_tank(x1=-dx/2, y1=-dx/2,
                                    x2=tank_length + dx/2, y2=tank_height+dx/2,
                                    dx=dx)

    x = numpy.concatenate( (xt1, xt2) )
    y = numpy.concatenate( (yt1, yt2) )

    h = numpy.ones_like(x) * h0
    m = numpy.ones_like(x) * ro*dx*dx*0.5
    rho = numpy.ones_like(x) * ro
    cs = numpy.ones_like(x) * co

    tank = base.get_particle_array(cl_precision="single", name="tank",
                                   type=Solid, x=x,y=y,m=m,rho=rho,h=h,cs=cs)
    np = tank.get_number_of_particles()

    # create the gate
    y1 = numpy.arange(dx/2, tank_height+1e-4, dx/2)
    x1 = numpy.ones_like(y1)*(0.38-dx/2)

    y2 = numpy.arange(dx/2+dx/4, tank_height+1e-4, dx/2)
    x2 = numpy.ones_like(y2)*(0.38-dx)

    y3 = numpy.arange(dx/2, tank_height+1e-4, dx/2)
    x3 = numpy.ones_like(y3)*(0.38-1.5*dx)

    x = numpy.concatenate( (x1, x2, x3) )
    y = numpy.concatenate( (y1, y2, y3) )

    h = numpy.ones_like(x) * h0
    m = numpy.ones_like(x) * 0.5 * dx/2 * dx/2 * ro
    rho = numpy.ones_like(x) * ro
    cs = numpy.ones_like(x) * co
    v = numpy.ones_like(x) * 1.5

    gate = base.get_particle_array(cl_precision="single", name="gate",
                                   x=x, y=y, m=m, rho=rho, h=h, cs=cs,
                                   v=v,
                                   type=Solid)

    np += gate.get_number_of_particles()
    print "Number of solid particles = %d"%(np)

    return [tank, gate]

def get_fluid_particles():
    
    # create the dam
    xf1, yf1 = geom.create_2D_filled_region(x1=dx, y1=dx,
                                            x2=0.38-2*dx,
                                            y2=0.15,
                                            dx=dx)

    xf2, yf2 = geom.create_2D_filled_region(x1=dx/2, y1=dx/2,
                                            x2=0.38-2*dx,
                                            y2=0.15,
                                            dx=dx)

    # create the bed
    xf3, yf3 = geom.create_2D_filled_region(x1=0.38+dx/2, y1=dx/2,
                                            x2=tank_length-dx, y2=d,
                                            dx=dx)

    xf4, yf4 = geom.create_2D_filled_region(x1=0.38, y1=dx,
                                            x2=tank_length-dx/2, y2=d,
                                            dx=dx)
    

    x = numpy.concatenate( (xf1, xf2, xf3, xf4) )
    y = numpy.concatenate( (yf1, yf2, yf3, yf4) )

    hf = numpy.ones_like(x) * h0
    mf = numpy.ones_like(x) * dx * dx * ro * 0.5
    rhof = numpy.ones_like(x) * ro
    csf = numpy.ones_like(x) * co
    rhop = numpy.ones_like(x) * ro
    
    fluid = base.get_particle_array(cl_precision="single",
                                    name="fluid", type=Fluid,
                                    x=x, y=y, h=hf, m=mf, rho=rhof,
                                    cs=csf, rhop=rhop)

    np = fluid.get_number_of_particles()
    print "Number of fluid particles = %d"%(np)

    return fluid

def get_particles(**args):
    fluid = get_fluid_particles()
    tank, gate = get_boundary_particles()

    return [fluid, tank, gate]

app = solver.Application()
s = solver.Solver(dim=2, integrator_type=solver.PredictorCorrectorIntegrator)

kernel = base.CubicSplineKernel(dim=2)

# define the artificial pressure term for the momentum equation
deltap = -1/1.3
n = 4

# pilot rho
s.add_operation(solver.SPHOperation(

    sph.ADKEPilotRho.withargs(h0=h0),
    on_types=[base.Fluid], from_types=[base.Fluid, base.Solid],
    updates=['rhop'], id='adke_rho'),

                )

# smoothing length update
s.add_operation(solver.SPHOperation(

    sph.ADKESmoothingUpdate.withargs(h0=h0, k=0.7, eps=0.5, hks=False),
    on_types=[base.Fluid], updates=['h'], id='adke'),
                
                )

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
                                  deltap=deltap, n=n),
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

s.add_operation(solver.SPHIntegration(

    sph.PositionStepping.withargs(),
    on_types=[base.Fluid,base.Solid],
    updates=["x","y"],
    id="step")

                )

s.add_operation(solver.SPHIntegration(

    sph.XSPHCorrection.withargs(),
    on_types=[base.Fluid,], from_types=[base.Fluid,],
    updates=["x","y"],
    id="xsph")

                )

dt = 1.25e-4

s.set_final_time(1.5)
s.set_time_step(dt)

app.setup(
    solver=s,
    variable_h=False, create_particles=get_particles, min_cell_size=4*h0,
    locator_type=base.NeighborLocatorType.SPHNeighborLocator,
    domain_manager=base.DomainManagerType.DomainManager,
    cl_locator_type=base.OpenCLNeighborLocatorType.AllPairNeighborLocator
    )

# this tells the solver to compute the max time step dynamically
s.time_step_function = solver.ViscousTimeStep(co=co,cfl=0.3,
                                              particles=s.particles)

app.run()
