""" An example Script to study the behavior of dynamic boundary
particles.

Dynamic boundary particles are boundary particles that obey all the
governing differential equations but their positions are fixed or
pre determined.

The setup is described as Test 1 of "Boundary Conditions Generated by
Dynamic Particles in SPH Methods" by A.J.C. Crespo and
M. Gomez-Gesteria and R.A. Dalrymple, CMC, vol 5, no 3 pp 173-184

Setup:
------
                    o [0, 0.3]
                    



        x   x   x   x   x   x   x                    
          x   x   x   x   x   x     
          -----     
           dx 


 o -- fluid particle
 x -- boundary particles


Y
|
|   Z
|  /
| /
|/_______X

Boundary particles are placed on a staggered grid with a bias of dx/2

The fluid particle falls under the influence of gravity and interacts
with the boundary particles. Since the boundary particles take part in
the equations of motion, the density of of the fluid and the boundary
particles is expected to increase when the fluid particle 'sees' the
boundary particle.

The increase in density causes a corresponding increase in pressure which 
causes a repulsive mechanism when put in the momentum equation.

Behavior:
---------
We study the motion of the fluid particle in this simple configuration. 
From the output files, observe the motion (`x` vs `y`) of the particle.

A state space plot of Velocity (`v`) V/S Position (`y`) should ideally
be a closed loop implying the conservation of energy. 

An alternative setup could be switching off gravity and imposing an
initial velocity on the particle directed towards the boundary. We can
study the ability of the method to prevent penetration by observing
the minimum distance 'y' from the wall for increasing velocities.

Parameters:
-----------

The maximum velocity is estimated as Vmax = sqrt(2*9.81*0.3) and the
numerical sound speed is taken as 10*Vmax ~ 25.0 m/s

The reference density is taken as 1.0

h = 2.097e-2
dx = dy = h/(1.3)
g = -9.81

"""

import logging, numpy
import sys

import pysph.solver.api as solver
import pysph.sph.api as sph
import pysph.base.api as base

Fluid = base.ParticleType.Fluid
Solid = base.ParticleType.Solid

fname = sys.argv[0][:-3]
app = solver.Application(fname=fname)

#global variables
h = 2.097e-2
dx = dy = h/(1.3)
g = -9.81

#define the fluid particle
xf = numpy.array([0])
yf = numpy.array([0.3])
hf = numpy.array([h])
mf = numpy.array([1.0])
vf = numpy.array([0.0])
cf = numpy.array([25.0])
rhof = numpy.array([1.0])

#define the staggered grid of boundary particles
xb = numpy.array([-dx, 0, dx, -dx/2, dx/2])
yb = numpy.array([0, 0, 0, -dy/2, -dy/2])
hb = numpy.ones_like(xb) * h
mb = numpy.ones_like(xb)
rhob = numpy.ones_like(xb)

fluid = base.get_particle_array(name="fluid", type=0, x=xf, y=yf,
                                h=hf, m=mf, rho=rhof, v=vf, cf=cf)

boundary = base.get_particle_array(name="boundary", type=1, x=xb, y=yb, 
                                   h=hb, m=mb, rho=rhob)

particles = base.Particles(arrays=[boundary,fluid])
app.particles = particles

s = solver.Solver(dim=2, integrator_type=solver.EulerIntegrator)

#Equation of state
s.add_operation(solver.SPHOperation(
        
        sph.TaitEquation.withargs(co=25.0, ro=1.0), 
        on_types=[Fluid, Solid], 
        updates=['p', 'cs'],
        id='eos')

                )

#Continuity equation
s.add_operation(solver.SPHIntegration(
            
            sph.SPHDensityRate.withargs(), 
            from_types=[Fluid, Solid], on_types=[Fluid, Solid],
            updates=['rho'], id='density')

                )

#momentum equation, no viscosity
s.add_operation(solver.SPHIntegration(

    sph.MomentumEquation.withargs(alpha=0.0, beta=0.0),
    on_types=[Fluid], from_types=[Fluid, Solid],  
    updates=['u','v'], id='mom')

                )

#Gravity force
s.add_operation(solver.SPHIntegration(
        
         sph.GravityForce.withargs(gy=-9.81),
         on_types=[Fluid],
         updates=['u','v'],id='gravity')
                 
                 )

#XSPH correction
s.add_operation(solver.SPHIntegration(
        
        sph.XSPHCorrection.withargs(eps=0.1), 
        on_types=[Fluid],
        from_types=[Fluid],
        updates=['x','y'], id='xsph')
                
                )

#Position stepping
s.add_operation(solver.SPHIntegration(

        sph.PositionStepping.withargs(), 
        on_types=[Fluid], 
        updates=['x','y'], id='step')
                
                )

s.set_final_time(1)
s.set_time_step(3e-4)

app.set_solver(s)

app.run()
