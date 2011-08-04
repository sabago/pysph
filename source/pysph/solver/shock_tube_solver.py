""" A simple shock tube solver """

from optparse import OptionGroup, Option
import numpy

import pysph.base.api as base
import pysph.sph.api as sph
from solver import Solver
from sph_equation import SPHOperation, SPHIntegration

Fluids = base.ParticleType.Fluid
Solids = base.ParticleType.Solid
Boundary = base.ParticleType.Boundary

def standard_shock_tube_data(name="", type=0, cl_precision="double",
                             nl=320, nr=80, smoothing_length=None, **kwargs):
    """ Standard 400 particles shock tube problem """
    
    dxl = 0.6/nl
    dxr = dxl*4
    
    x = numpy.ones(nl+nr)
    x[:nl] = numpy.arange(-0.6, -dxl+1e-10, dxl)
    x[nl:] = numpy.arange(dxr, 0.6+1e-10, dxr)

    m = numpy.ones_like(x)*dxl
    h = numpy.ones_like(x)*2*dxr

    if smoothing_length:
        h = numpy.ones_like(x) * smoothing_length
    
    rho = numpy.ones_like(x)
    rho[nl:] = 0.25
    
    u = numpy.zeros_like(x)
    
    e = numpy.ones_like(x)
    e[:nl] = 2.5
    e[nl:] = 1.795

    p = 0.4*rho*e

    cs = numpy.sqrt(1.4*p/rho)

    idx = numpy.arange(nl+nr)
    
    return base.get_particle_array(name=name,x=x,m=m,h=h,rho=rho,p=p,e=e,
                                   cs=cs,type=type, idx=idx,
                                   cl_precision=cl_precision)

class ShockTubeSolver(Solver):

    def get_options(self, opt_parser):

        self.defaults = dict(alpha=1.0,
                             beta=1.0)

        opt = OptionGroup(opt_parser, "Shock tube problem options")

        opt.add_option("--alpha", action="store", type="float",
                       dest="alpha", default=self.defaults["alpha"],
                       help="Set the artificial viscosity parameter alpha")

        opt.add_option("--beta", action="store", type="float",
                       dest="beta", default=self.defaults["alpha"],
                       help="Set the artificial viscosity parameter beta")

        return opt
    
    def setup_solver(self, options=None):

        options = options or self.defaults
        alpha = options.get("alpha")
        beta = options.get("beta")
        hks = options.get("hks")

        #create the sph operation objects

        self.add_operation(SPHOperation(

            sph.SPHRho.withargs(hks=hks),
            from_types=[Fluids], on_types=[Fluids], 
            updates=['rho'], id = 'density')

                           )

        self.add_operation(SPHOperation(

            sph.IdealGasEquation.withargs(),
            on_types = [Fluids],
            updates=['p', 'cs'], id='eos')

                           )

        self.add_operation(SPHIntegration(

            sph.MomentumEquation.withargs(alpha=alpha, beta=beta, hks=hks),
            from_types=[Fluids], on_types=[Fluids], 
            updates=['u'], id='mom')

                           )
        
        self.add_operation(SPHIntegration(

            sph.EnergyEquation.withargs(hks=hks),
            from_types=[Fluids],
            on_types=[Fluids], updates=['e'], id='enr')

                           )

        # Indicate that stepping is only needed for Fluids

        self.add_operation_step([Fluids])

#############################################################################
class ADKEShockTubeSolver(Solver):

    def __init__(self, dim, integrator_type, h0, eps, k, g1, g2, alpha, beta,
                 kernel = base.CubicSplineKernel, hks=False):

        # solver parameters
        self.h0 = h0
        self.adke_eps = eps
        self.k = k
        self.g1 = g1
        self.g2 = g2
        self.alpha = alpha
        self.beta = beta

        # Hernquist and Katz normalization
        self.hks = hks

        # the SPH kernel to use
        self.kernel = kernel(dim)

        self.defaults = dict(alpha=alpha,
                             beta=beta,
                             adke_eps=eps,
                             adke_k=k,
                             adke_h0=h0,
                             g1=g1,
                             g2=g2)                             

        # base class constructor
        Solver.__init__(self, dim, integrator_type)

    def get_options(self, opt_parser):

        opt = OptionGroup(opt_parser, "ADKEShockTubeSolver options")

        opt.add_option("--alpha", action="store", type="float",
                       dest="alpha", default=self.defaults["alpha"],
                       help="Set the artificial viscosity parameter alpha")

        opt.add_option("--beta", action="store", type="float",
                       dest="beta", default=self.defaults["alpha"],
                       help="Set the artificial viscosity parameter beta")

        opt.add_option("--adke-eps", action="store", type="float",
                       dest="adke_eps", default=self.defaults.get("adke_eps"),
                       help="Sensitivity parameter eps for the ADKE pocedure")

        opt.add_option("--adke-k", action="store", type="float",
                       dest="adke_k", default=self.defaults.get("adke_k"),
                       help="Scaling parameter k for the ADKE pocedure")

        opt.add_option("--adke-h0", action="store", type="float",
                       dest="adke_h0", default=self.defaults.get("adke_h0"),
                       help="Initial smoothing length h0 for the ADKE pocedure")

        opt.add_option("--g1", action="store", type="float",
                       dest="g1", default=self.defaults.get("g1"),
                       help="Artificial heating term coefficient g1")

        opt.add_option("--g2", action="store", type="float",
                       dest="g2", default=self.defaults.get("g2"),
                       help="Artificial heating term coefficient g2")

        return opt        
    
    def setup_solver(self, options=None):

        options = options or self.defaults

        hks = options.get("hks")
        kernel = self.kernel

        # ADKE parameters
        h0 = options.get("adke_h0")
        eps = options.get("adke_eps")
        k = options.get("adke_k")

        # Artificial heat parameters
        g1 = options.get("g1")
        g2 = options.get("g2")

        # Artificial viscosity parameters
        alpha = options.get("alpha")
        beta = options.get("beta")

        self.add_operation(SPHOperation(

            sph.ADKEPilotRho.withargs(h0=h0),
            on_types=[Fluids], from_types=[Fluids,Boundary],
            updates=['rhop'], id='adke_rho'),

                        )

        # smoothing length update
        self.add_operation(SPHOperation(
            
            sph.ADKESmoothingUpdate.withargs(h0=h0, k=k, eps=eps, hks=hks),
            on_types=[Fluids], updates=['h'], id='adke'),
                        
                        )

        # summation density
        self.add_operation(SPHOperation(
            
            sph.SPHRho.withargs(hks=hks),
            from_types=[Fluids, Boundary], on_types=[Fluids], 
            updates=['rho'], id = 'density')
                        
                        )

        # ideal gas equation
        self.add_operation(SPHOperation(
            
            sph.IdealGasEquation.withargs(),
            on_types = [Fluids], updates=['p', 'cs'], id='eos')
                        
                        )

        # velocity divergence
        self.add_operation(SPHOperation(
            
            sph.VelocityDivergence.withargs(hks=hks),
            on_types=[Fluids], from_types=[Fluids, Boundary],
            updates=['div'], id='vdivergence'),
                        
                    )

        #conduction coefficient update
        self.add_operation(SPHOperation(
            
            sph.ADKEConductionCoeffUpdate.withargs(g1=g1, g2=g2),
            on_types=[Fluids],
            updates=['q'], id='qcoeff'),
                        
                        )

        # momentum equation
        self.add_operation(SPHIntegration(
    
            sph.MomentumEquation.withargs(alpha=alpha, beta=beta, hks=hks),
            from_types=[Fluids, Boundary], on_types=[Fluids], 
            updates=['u'], id='mom')
                        
                        )

        # energy equation
        self.add_operation(SPHIntegration(
            
            sph.EnergyEquation.withargs(hks=hks,),
            from_types=[Fluids, Boundary],
            on_types=[Fluids], updates=['e'], id='enr')
                        
                        )

        # artificial heat 
        self.add_operation(SPHIntegration(
            
            sph.ArtificialHeat.withargs(eta=0.1, hks=hks),
            on_types=[Fluids], from_types=[Fluids,Boundary],
            updates=['e'], id='aheat'),
                        
                        )
        
        # position step
        self.add_operation_step([Fluids])
        
        
        
        
    
        
        
