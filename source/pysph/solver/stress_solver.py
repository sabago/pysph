""" An example solver for the circular patch of fluid """

import numpy

from optparse import OptionGroup, Option

import pysph.base.api as base

import pysph.sph.api as sph
from pysph.sph.funcs import stress_funcs
from pysph.sph.funcs import eos_funcs
from pysph.sph.funcs import viscosity_funcs

from solver import Solver
from post_step_functions import CFLTimeStepFunction
from sph_equation import SPHOperation, SPHIntegration
from pysph.sph.funcs.arithmetic_funcs import PropertyGet, PropertyAdd
from pysph.sph.funcs.basic_funcs import KernelSum

Fluids = base.ParticleType.Fluid
Solids = base.ParticleType.Solid


def get_particle_array(xsph=True, mart_stress=True, **kwargs):
    kwargs.setdefault('type', 1)
    kwargs.setdefault('name', 'solid')
    
    pa = base.get_particle_array(**kwargs)
    
    for i in range(3):
        for j in range(i+1):
            pa.add_property(dict(name='sigma%d%d'%(j,i)))

    if xsph:
        pa.add_property(dict(name='ubar'))
        pa.add_property(dict(name='vbar'))
        pa.add_property(dict(name='wbar'))

    if mart_stress:        
        for i in range(3):
            for j in range(i+1):
                pa.add_property(dict(name='MArtStress%d%d'%(j,i)))

    return pa


def get_circular_patch(name="", type=1, dx=0.25):
    
    x,y = numpy.mgrid[-1.05:1.05+1e-4:dx, -1.05:1.05+1e-4:dx]
    x = x.ravel()
    y = y.ravel()
 
    m = numpy.ones_like(x)*dx*dx
    h = numpy.ones_like(x)*2*dx
    rho = numpy.ones_like(x)
    z = 1-rho

    p = 0.5*1.0*100*100*(1 - (x**2 + y**2))

    cs = numpy.ones_like(x) * 100.0

    u = -100*x
    v = 100*y

    indices = []

    for i in range(len(x)):
        if numpy.sqrt(x[i]*x[i] + y[i]*y[i]) - 1 > 1e-10:
            indices.append(i)
            
    pa = base.get_particle_array(x=x, y=y, m=m, rho=rho, h=h, p=p, u=u, v=v,
                                 cs=cs,name=name, type=type,
                                 sigma00=z, sigma11=z, sigma22=z,
                                 sigma01=z, sigma12=z, sigma02=z)
    pa.constants['E'] = 1e9
    pa.constants['nu'] = 0.3
    pa.constants['G'] = pa.constants['E']/(2.0*1+pa.constants['nu'])
    pa.constants['K'] = stress_funcs.get_K(pa.constants['G'], pa.constants['nu'])
    pa.constants['rho0'] = 1.0
    pa.constants['c_s'] = numpy.sqrt(pa.constants['K']/pa.constants['rho0'])
            
    la = base.LongArray(len(indices))
    la.set_data(numpy.array(indices))

    pa.remove_particles(la)

    pa.set(idx=numpy.arange(len(pa.x)))
 
    print 'Number of particles: ', len(pa.x)
    
    return pa

class StressSolver(Solver):
    def __init__(self, dim, integrator_type, xsph=0.5, marts_eps=0.3, marts_n=4,
                 CFL=None, martv_alpha=1.0, martv_beta=1.0,
                 co=None, ro=None):
        ''' constructor
        
        Parameters
        ----------

        xsph : float
            correction factor for xsph (0=disabled, default=0.5)

        marts_eps : float
            correction factor epsilon for Monaghan's artificial stress term
            (0=disabled, default=0.3)

        marts_n : float
            correction factor kernel exponent for Monaghan's
            artificial stress term

        CFL : float or None
            the CFL number if time-step is to be based on CFL (use < 0.3)

        dim, integrator_type : see :py:meth:`Solver.__init__`

        '''
        self.defaults = dict(xsph=xsph,
                             marts_eps=marts_eps,
                             marts_n=marts_n,
                             martv_alpha=martv_alpha,
                             martv_beta=martv_beta,
                             cfl=CFL,
                             co=co,
                             ro=ro
                             )
        Solver.__init__(self, dim, integrator_type)

    def initialize(self):
        Solver.initialize(self)
        self.print_properties.append('sigma00')
        self.print_properties.extend(['sigma01', 'sigma11'])
        self.print_properties.extend(['sigma02', 'sigma12', 'sigma22'])
        self.print_properties.append('MArtStress00')
        self.print_properties.extend(['MArtStress01', 'MArtStress11'])
        self.print_properties.extend(['MArtStress02', 'MArtStress12', 'MArtStress22'])

    def get_options(self, opt_parser):
        opt = OptionGroup(opt_parser, "Stress Solver Options")
        
        opt.add_option('--xsph', action='store', type='float',
                       dest='xsph', default=self.defaults['xsph'],
                       help='set the XSPH correction weight factor (default=0.5)')
        opt.add_option('--marts_eps', dest='marts_eps', type='float',
                       default=self.defaults['marts_eps'],
                       help='set the Monaghan artificial stress weight factor (0.3)')
        opt.add_option('--marts_n', dest='marts_n', type='float',
                       default=self.defaults['marts_n'],
                       help='set the Monaghan artificial stress exponent (4)')

        opt.add_option('--martv_alpha', dest='martv_alpha', type='float',
                       default=self.defaults['martv_alpha'],
                       help='set the Monaghan artificial viscosity alpha (1)')
        
        opt.add_option('--martv_beta', dest='martv_beta', type='float',
                       default=self.defaults['martv_beta'],
                       help='set the Monaghan artificial viscosity beta (1)')

        opt.add_option('--co', dest="co", type="float",
                       default=self.defaults["co"],
                       help="Set the reference sound speed c0 ")

        opt.add_option("--ro", dest="ro", type="float",
                       default=self.defaults["ro"],
                       help="Set the reference density r0")
        
        cfl_opt = Option('--cfl', dest='cfl', type='float',
                         default=self.defaults['cfl'],
                         help='set the cfl number for determining the timestep '
                         'of simulation')

        return opt, cfl_opt

    def setup_solver(self, options=None):
        
        options = options or self.defaults
        xsph = options.get('xsph')
        marts_eps = options.get('marts_eps')
        marts_n = options.get('marts_n')
        martv_alpha = options.get('martv_alpha')
        martv_beta = options.get('martv_beta')

        cfl = options.get('cfl')

        co = options.get("co")
        ro = options.get("ro")

        #create the sph operation objects
        self.add_operation(SPHOperation(
                stress_funcs.BulkModulusPEqn,
                on_types=[Solids],
                updates=['p'],
                id='eos')
            )

        if xsph:
            self.add_operation(SPHOperation(
                
                sph.XSPHCorrection.withargs(eps=xsph, hks=False),
                from_types=[Solids], on_types=[Solids],
                updates=['ubar','vbar','wbar'],
                id='xsphcorr')

                               )
        # Monaghan Artificial Stress operations
        if marts_eps:
            self.add_operation(SPHOperation(
                
                stress_funcs.MonaghanArtStressD.withargs(eps=marts_eps),
                on_types=[Solids],
                updates=['MArtStress00','MArtStress11','MArtStress22'],
                id='mart_stress_d')
                               )
            self.add_operation(SPHOperation(

                stress_funcs.MonaghanArtStressS.withargs(eps=marts_eps),
                on_types=[Solids],
                updates=['MArtStress12','MArtStress02','MArtStress01'],
                id='mart_stress_s')
                               )
            self.add_operation(SPHIntegration(
                
                stress_funcs.MonaghanArtStressAcc.withargs(n=marts_n),
                from_types=[Fluids, Solids], on_types=[Solids],
                updates=['u','v','w'],
                id='mart_stressacc')
                               )

        if xsph:
            self.add_operation(SPHIntegration(

                sph.XSPHDensityRate.withargs(hks=False),
                from_types=[Solids], on_types=[Solids],
                updates=['rho'],
                id='density')
                               )
        else:
            self.add_operation(SPHIntegration(

                sph.SPHDensityRate.withargs(hks=False),
                from_types=[Solids], on_types=[Solids],
                updates=['rho'],
                id='density')
                               )

        self.add_operation(SPHIntegration(

            stress_funcs.SimpleStressAcceleration,
            from_types=[Fluids, Solids], on_types=[Solids],
            updates=['u','v','w'],
            id='stressacc')
                           )

        self.add_operation(SPHIntegration(

            stress_funcs.PressureAcceleration.withargs(alpha=martv_alpha,
                                                       beta=martv_beta,
                                                       eta=0.0),
            from_types=[Fluids, Solids], on_types=[Solids],
            updates=['u','v','w'],
            id='pacc')
                           )

        self.add_operation(SPHIntegration(

            stress_funcs.StressRateD.withargs(xsph=bool(xsph)),
            from_types=[Fluids, Solids], on_types=[Solids],
            updates=['sigma00','sigma11','sigma22'],
            id='stressD')
                           )

        self.add_operation(SPHIntegration(
            
            stress_funcs.StressRateS.withargs(xsph=bool(xsph)),
            from_types=[Fluids, Solids], on_types=[Solids],
            updates=['sigma12','sigma02','sigma01'],
            id='stressS')
                           )
        

        # position stepping
        self.add_operation(SPHIntegration(

            sph.PositionStepping,
            on_types=[Solids],
            updates=['x','y','z'],
            id='pos')
                           )

        # xsph correction to position stepping
        if xsph:
            self.add_operation(SPHIntegration(
                PropertyGet.withargs(prop_names=['ubar','vbar','wbar'][:self.dim]),
                on_types=[Solids],
                updates=['x','y','z'][:self.dim],
                id='xsph_pos')
                               )
            
        #self.add_operation_step([Solids])
        #self.add_operation_xsph(eps=0.5, hks=False)

        if cfl:
            self.pre_step_functions.append(CFLTimeStepFunction(cfl))


    def _setup_solver(self, options=None):
        E = 1e7
        nu = 0.3975
        K = 1.0/3.0 * E/(1-2*nu)
        G = 0.5*E/(1+nu)
        ro = 1.0
        co = numpy.sqrt(K/ro)

        deltap = 0.001
        fac = 1e-10

        print "Using ", G, co
        s=self

        # Velocity Gradient tensor
        s.add_operation(SPHOperation(
            
            sph.VelocityGradient2D.withargs(), on_types=[Solids,],
            id="vgrad")

                        )
        # Equation of state
        s.add_operation(SPHOperation(
            
            sph.IsothermalEquation.withargs(ro=ro, co=co), on_types=[Solids,],
            id="eos", updates=['p'])

                        )
        # density rate
        s.add_operation(SPHIntegration(
            
            sph.SPHDensityRate.withargs(), on_types=[Solids,], from_types=[Solids],
            id="density", updates=['rho'])
                        
                        )

        # momentum equation
        s.add_operation(SPHIntegration(
            
            sph.MomentumEquationWithStress2D.withargs(deltap=deltap,
                                                      n=4.0, epsp=0.3,
                                                      theta_factor=1e-10),
            on_types=[Solids,],
            from_types=[Solids,], id="momentum", updates=['u','v'])
                    
                        )
        
        # momentum equation artificial viscosity
        s.add_operation(SPHIntegration(
            
            sph.MonaghanArtificialVsicosity.withargs(alpha=1.0, beta=1.0),
            on_types=[Solids,], from_types=[Solids,],
            id="avisc", updates=['u','v'])
                        
                        )
        # stress rate
        s.add_operation(SPHIntegration(
            
            sph.HookesDeviatoricStressRate2D.withargs(shear_mod=G),
            on_types=[Solids,],
            id="stressrate")
                        
                        )
        # position stepping
        s.add_operation(SPHIntegration(
            
            sph.PositionStepping.withargs(),
            on_types=[Solids,],
            id="step", updates=['x','y'])
                        
                        )

#############################################################################
