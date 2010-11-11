""" An example Script to study the behavior of Monaghan type repulsive
particles (Smoothed Particle Hydrodynamics, Reports on Progresses in
Physics)

The boundary particles are an improvement over the Lenard Jones type
repulsive boundary particles. One of the main features is that a
particle moving parallel to the wall will experience the same force.

The force exerted on a boundary particle is 

f = f1(x)*f2(y) nk

where f1 is a function of the component of the projection of the
vector rab onto the tangential direction and f2 is a function of the
component of the normal projection of rab.

Each boundary particle must have therefore an associated normal and
tangent. 

The setup is described as Test 1 of "Boundary Conditions Generated by
Dynamic Particles in SPH Methods" by A.J.C. Crespo and
M. Gomez-Gesteria and R.A. Dalrymple, CMC, vol 5, no 3 pp 173-184

Setup:
------
                    o [0, 0.3]
                    



        x   x   x   x   x   x   x                    
          -----     
           dp 


 o -- fluid particle
 x -- boundary particles


Y
|
|   Z
|  /
| /
|/_______X


The fluid particle falls under the influence of gravity and interacts
with the boundary particles. When the particle `sees` the boundary
particle for the interaction of the boundary force term, a repulsion
is activated on the fluid particle.

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


Running:
--------
run like so:
python monaghanbc.py --freq <print-freq> --directory ./monaghanbc

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
app.process_command_line()

#global variables
h = 2.097e-2
dx = dy = h/(1.3)
g = -9.81

xf = numpy.array([0])
yf = numpy.array([0.3])
hf = numpy.array([h])
mf = numpy.array([1.0])
vf = numpy.array([0.0])
cf = numpy.array([25.0])
rhof = numpy.array([1.0])

fluid = base.get_particle_array(name="fluid", type=Fluid, x=xf, y=yf,
                                h=hf, m=mf, rho=rhof, v=vf, cs=cf)

#generate the boundary
l = base.Line(base.Point(-.5), 1.0, 0)
g = base.Geometry('line', [l], False)
g.mesh_geometry(dx)
boundary = g.get_particle_array(re_orient=True)

boundary.m[:] = 1.0

particles = base.Particles(arrays=[fluid, boundary])
app.particles = particles
s = solver.Solver(base.HarmonicKernel(dim=2, n=3), solver.RK4Integrator)


#Tait equation
s.add_operation(solver.SPHAssignment(
        
        sph.TaitEquation(co=25.0, ro=1.0), 
        on_types=[Fluid], 
        updates=['p','cs'],
        id='eos')

                )

#continuity equation
s.add_operation(solver.SPHSummationODE(
            
            sph.SPHDensityRate(), from_types=[Fluid], 
            on_types=[Fluid],
            updates=['rho'], id='density')

                )

#momentum equation
s.add_operation(solver.SPHSummationODE(

    sph.MomentumEquation(alpha=0.0, beta=0.0,),
    on_types=[Fluid], 
    from_types=[Fluid], 
    updates=['u','v'], id='mom')

                )

#gravity force
s.add_operation(solver.SPHSimpleODE(
        
         sph.GravityForce(gy=-9.81),
         on_types=[Fluid],
         updates=['u','v'],id='gravity')
                 
                 )

#the boundary force
s.add_operation(solver.SPHSummationODE(
        
        sph.MonaghanBoundaryForce(delp=dx),
        on_types=[Fluid], from_types=[Solid], updates=['u','v'],
        id='bforce')
                
                )

#xsph correction
s.add_operation(solver.SPHSummationODE(
        
        sph.XSPHCorrection(eps=0.1),
        from_types=[Fluid],
        on_types=[Fluid],  updates=['x','y'], id='xsph')
                
                )

#Position stepping
s.add_operation(solver.SPHSimpleODE(

        sph.PositionStepping(), 
        on_types=[Fluid], 
        updates=['x','y'], id='step')
                
                )

s.set_final_time(1)
s.set_time_step(3e-4)

app.set_solver(s)

app.run()