""" Balls colliding in 2D """

import numpy

import pysph.base.api as base
import pysph.sph.api as sph
import pysph.solver.api as solver

app = solver.Application()

Solid = base.ParticleType.Solid

E = 1e7
nu = 0.3975
G = E/(2.0*(1+nu))
K = sph.get_K(G, nu)
ro = 1.0
co = numpy.sqrt(K/ro)

def create_particles(two_arr=False):
    #x,y = numpy.mgrid[-1.05:1.05+1e-4:dx, -0.105:0.105+1e-4:dx]
    dx = 0.001 # 1mm
    ri = 0.03 # 3cm inner radius
    ro = 0.04 # 4cm outer radius
    spacing = 0.041 # spacing = 2*5cm
    
    x,y = numpy.mgrid[-ro:ro:dx, -ro:ro:dx]
    x = x.ravel()
    y = y.ravel()
    
    d = (x*x+y*y)
    keep = numpy.flatnonzero((ri*ri<=d) * (d<ro*ro))
    x = x[keep]
    y = y[keep]

    print 'num_particles', len(x)*2
    
    if not two_arr:
        x = numpy.concatenate([x-spacing,x+spacing])
        y = numpy.concatenate([y,y])

    #print bdry, numpy.flatnonzero(bdry)
    m = numpy.ones_like(x)*dx*dx
    h = numpy.ones_like(x)*1.4*dx
    rho = numpy.ones_like(x)
    z = numpy.zeros_like(x)

    p = 0.5*1.0*100*100*(1 - (x**2 + y**2))

    cs = numpy.ones_like(x) * 10000.0

    # u is set later
    v = z
    u_f = 0.059

    p *= 0
    h *= 1

    pa = base.get_particle_array(cl_precision="single",
                                 name="ball", type=Solid,
                                 x=x+spacing, y=y,
                                 m=m, rho=rho, h=h,
                                 p=p, cs=cs,
                                 u=z, v=v)

    pa.cs[:] = co
    pa.u = pa.cs*u_f*(2*(x<0)-1)

    return pa

s = solver.Solver(dim=2, integrator_type=solver.PredictorCorrectorIntegrator)


# Add the operations

# Velocity Gradient tensor
s.add_operation(solver.SPHOperation(

    sph.VelocityGradient2D.withargs(), on_types=[Solid,],
    id="vgrad")

                )

# Equation of state
s.add_operation(solver.SPHOperation(

    sph.IsothermalEquation.withargs(ro=ro, co=ro), on_types=[Solid,],
    id="eos", updates=['p'])

                )

# density rate
s.add_operation(solver.SPHIntegration(

    sph.SPHDensityRate.withargs(), on_types=[Solid,], from_types=[Solid],
    id="density", updates=['rho'])

                )

# momentum equation
s.add_operation(solver.SPHIntegration(

    sph.MomentumEquationWithStress2D.withargs(), on_types=[Solid,],
    from_types=[Solid,], id="momentum", updates=['u','v'])

                )

# momentum equation artificial viscosity
s.add_operation(solver.SPHIntegration(

    sph.MonaghanArtificialVsicosity.withargs(alpha=1.0, beta=1.0),
    on_types=[Solid,], from_types=[Solid,],
    id="avisc", updates=['u','v'])

                )

# stress rate
s.add_operation(solver.SPHIntegration(

    sph.HookesDeviatoricStressRate2D.withargs(shear_mod=G),
    on_types=[Solid,],
    id="stressrate")

                )

# position stepping
s.add_operation(solver.SPHIntegration(

    sph.PositionStepping.withargs(),
    on_types=[Solid,],
    id="step", updates=['x','y'])
    
                )

# XSPH
s.add_operation(solver.SPHIntegration(

    sph.XSPHCorrection.withargs(eps=0.5),
    on_types=[Solid,], from_types=[Solid,],
    id="xsph", updates=['x','y'])

                )

app.set_solver(s, callable=create_particles)

dt = 1e-8
tf = 1e-2
s.set_time_step(dt)
s.set_final_time(tf)
s.set_kernel_correction(-1)
s.pfreq = 100

app.run()
