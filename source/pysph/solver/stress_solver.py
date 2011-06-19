""" An example solver for the circular patch of fluid """

import numpy

from optparse import OptionGroup

import pysph.base.api as base

import pysph.sph.api as sph
from pysph.sph.funcs import stress_funcs 

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
    def __init__(self, dim, integrator_type, xsph=0.5, mart_eps=0.3, mart_n=4,
                 CFL=None):
        ''' constructor
        
        Parameters
        ----------
        xsph : float
            correction factor for xsph (0=disabled, default=0.5)
        mart_eps : float
            correction factor epsilon for Monaghan's artificial stress term
            (0=disabled, default=0.3)
        mart_n : float
            correction factor kernel exponent for Monaghan's artificial stress term
        CFL : float or None
            the CFL number if time-step is to be based on CFL (use < 0.3)

        dim, integrator_type : see :py:meth:`Solver.__init__`

        '''
        self.xsph = xsph
        self.mart_eps = mart_eps
        self.mart_n = mart_n
        self.cfl = CFL
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
        
        opt.add_option('--xsph_fac', action='store',
                       dest='xsph_fac',
                       help='set the XSPH correction weight factor (default=0.5)')
        opt.add_option('--mart_eps', dest='mart_eps',
                       help='set the Monaghan artificial stress weight factor (0.3)')
        opt.add_option('--mart_n', dest='mart_n',
                       help='set the Monaghan artificial stress exponent (4)')
        return opt

    def setup_solver(self, options=None):
        
        options = options or {}
        xsph = options.get('xsph') or  self.xsph
        mart_eps = options.get('mart_eps') or self.mart_eps
        mart_n = options.get('mart_n') or self.mart_n
        cfl = options.get('cfl') or self.cfl

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

        if mart_eps:
            # Monaghan Artificial Stress operations
            self.add_operation(SPHOperation(
                    stress_funcs.MonaghanArtStressD.withargs(eps=mart_eps),
                    on_types=[Solids],
                    updates=['MArtStress00','MArtStress11','MArtStress22'],
                    id='mart_stress_d')
                               )
            
            self.add_operation(SPHOperation(
                    stress_funcs.MonaghanArtStressS.withargs(eps=mart_eps),
                    on_types=[Solids],
                    updates=['MArtStress12','MArtStress02','MArtStress01'],
                    id='mart_stress_s')
                               )
            
            self.add_operation(SPHIntegration(
                    stress_funcs.MonaghanArtStressAcc.withargs(n=mart_n),
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
                stress_funcs.PressureAcceleration,
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

        if xsph:
            # xsph correction to position stepping
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

#############################################################################
