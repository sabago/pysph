""" Balls colliding in 2D """

import numpy

import pysph.base.api as base
import pysph.sph.api as sph
import pysph.solver.api as solver

import pysph.sph.funcs.stress_funcs as stress_funcs

app = solver.Application()

Solid = base.ParticleType.Solid

E = 1e7
nu = 0.3975
G = E/(2.0*(1+nu))
K = sph.get_K(G, nu)
ro = 1.0
co = numpy.sqrt(K/ro)

deltap = 0.001
fac=1e-10

print "co, ro, G = ", co, ro, G

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

    pa.constants['dr0'] = dx
    pa.constants["rho0"] = ro

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

    sph.IsothermalEquation.withargs(ro=ro, co=co), on_types=[Solid,],
    id="eos", updates=['p'])

                )

# Artificial stress
s.add_operation(solver.SPHOperation(

    sph.MonaghanArtificialStress.withargs(eps=0.3),
    on_types=[Solid,],
    id="art_stress",)

                )

# density rate
s.add_operation(solver.SPHIntegration(

    sph.SPHDensityRate.withargs(), on_types=[Solid,], from_types=[Solid],
    id="density", updates=['rho'])

                )

# momentum equation artificial viscosity
s.add_operation(solver.SPHIntegration(

    sph.MonaghanArtificialViscosity.withargs(alpha=1.0, beta=1.0),
    on_types=[Solid,], from_types=[Solid,],
    id="avisc", updates=['u','v'])

                )

# momentum equation
s.add_operation(solver.SPHIntegration(

    sph.MomentumEquationWithStress2D.withargs(deltap=deltap, n=4),
    on_types=[Solid,],
    from_types=[Solid,], id="momentum", updates=['u','v'])

                )

# s.add_operation(solver.SPHIntegration(
    
#     sph.MonaghanArtStressAcc.withargs(n=4, deltap=deltap, rho0=ro,
#                                       R="R_"),
#     from_types=[Solid], on_types=[Solid],
#     updates=['u','v'],
#     id='mart_stressacc')

#                  )

# XSPH
s.add_operation(solver.SPHIntegration(

    sph.XSPHCorrection.withargs(eps=0.5),
    on_types=[Solid,], from_types=[Solid,],
    id="xsph", updates=['u','v'])

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

app.setup(s, create_particles=create_particles)

dt = 1e-8
tf = 1e-2
s.set_time_step(dt)
s.set_final_time(tf)
s.set_kernel_correction(-1)
s.pfreq = 100

app.run()

###############################################################################
# DEBUG
s1 = solver.Solver(dim=2, integrator_type=solver.PredictorCorrectorIntegrator)
                   
# Velocity Gradient tensor
s1.add_operation(solver.SPHOperation(

    sph.VelocityGradient2D.withargs(), on_types=[Solid,],
    id="vgrad")

                )

# Equation of state
s1.add_operation(solver.SPHOperation(

    sph.IsothermalEquation.withargs(ro=ro, co=co), on_types=[Solid,],
    id="eos", updates=['p'])

                )

# density rate
s1.add_operation(solver.SPHIntegration(

    sph.SPHDensityRate.withargs(), on_types=[Solid,], from_types=[Solid],
    id="density", updates=['rho'])

                )


# s1.add_operation(solver.SPHOperation(
                
#     stress_funcs.MonaghanArtStressD.withargs(eps=0.3, stress="S_"),
#     on_types=[Solid],
#     updates=['MArtStress00','MArtStress11','MArtStress22'],
#     id='mart_stress_d')
#                    )

# s1.add_operation(solver.SPHOperation(

#     stress_funcs.MonaghanArtStressS.withargs(eps=0.3, stress="S_"),
#     on_types=[Solid],
#     updates=['MArtStress12','MArtStress02','MArtStress01'],
#     id='mart_stress_s')
#                  )

# s1.add_operation(solver.SPHIntegration(
    
#     stress_funcs.MonaghanArtStressAcc.withargs(n=4),
#     from_types=[Solid], on_types=[Solid],
#     updates=['u','v','w'],
#     id='mart_stressacc')
#                  )

# momentum equation
s1.add_operation(solver.SPHIntegration(

    sph.MomentumEquationWithStress2D.withargs(theta_factor=fac,
                                              deltap=deltap, n=4,
                                              epsp=0.3, epsm=0),
                                              
    on_types=[Solid,],
    from_types=[Solid,], id="momentum", updates=['u','v'])

                )

# s1.add_operation(solver.SPHIntegration(

#     stress_funcs.SimpleStressAcceleration.withargs(stress="S_"),
#     from_types=[Solid], on_types=[Solid],
#     updates=['u','v','w'],
#     id='stressacc')

#                  )


# momentum equation artificial viscosity
s1.add_operation(solver.SPHIntegration(

    sph.MonaghanArtificialVsicosity.withargs(alpha=1.0, beta=1.0, eta=0.0),
    on_types=[Solid,], from_types=[Solid,],
    id="avisc", updates=['u','v'])

                )

# stress rate
s1.add_operation(solver.SPHIntegration(

    sph.HookesDeviatoricStressRate2D.withargs(shear_mod=G),
    on_types=[Solid,],
    id="stressrate")

                )

# position stepping
s1.add_operation(solver.SPHIntegration(

    sph.PositionStepping.withargs(),
    on_types=[Solid,],
    id="step", updates=['x','y','z'])
    
                )

dt = 1e-8
tf = 1e-2
s1.set_time_step(dt)
s1.set_final_time(tf)
s1.set_kernel_correction(-1)
s1.pfreq = 100

app1.setup(s1, create_particles=create_particles)



#app.run()

# can be overriden by commandline arguments
dt = 1e-8
tf = 1e-2
s.set_time_step(dt)
s.set_final_time(tf)
s.set_kernel_correction(-1)
s.pfreq = 100

app2.setup(s, create_particles=create_particles)

#print [calc.id for calc in s.integrator.calcs]
#print [calc.id for calc in s1.integrator.calcs]

# particles = s.particles
# pa = particles.arrays[0]

def check():
    array1 = s.particles.arrays[0]
    array2 = s1.particles.arrays[0]

    props = ['x','y','u','v','rho','p']
    np = array1.get_number_of_particles()
    nk = array2.get_number_of_particles()

    assert np == nk

    for prop in props:
        p = array1.get(prop)
        k = array2.get(prop)

        err = abs(p - k)
        print prop, sum(err)/nk, max(err)

t = 0.0
while t < tf:

    print "Checkking at %g ", t
    check()
    print
    
    t += dt
    s.set_final_time(t)
    s1.set_final_time(t)

    s.solve(dt)
    s1.solve(dt)
