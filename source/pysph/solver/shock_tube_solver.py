""" A simple shock tube solver """

from optparse import OptionGroup, Option
import numpy

import pysph.base.api as base
import pysph.sph.api as sph
from solver import Solver
from sph_equation import SPHOperation, SPHIntegration

from integrator import GSPHIntegrator

Fluids = base.Fluid
Solids = base.Solid
Boundary = base.Boundary

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

############################################################################
# `ShockTubeSolver` class
############################################################################
class ShockTubeSolver(Solver):

    def __init__(self, dim, integrator_type, alpha=1.0, beta=1.0,
                 gamma=1.4, xsph_eps=0):

        self.dim = dim
        self.defaults = dict(alpha=alpha,
                             beta=beta,
                             gamma=gamma,
                             xsph_eps=xsph_eps)

        # base class constructor
        Solver.__init__(self, dim, integrator_type)

    def get_options(self, opt_parser):

        opt = OptionGroup(opt_parser, "ShockTubeSolver options")

        opt.add_option("--alpha", action="store", type="float",
                       dest="alpha", default=self.defaults["alpha"],
                       help="Set the artificial viscosity parameter alpha")

        opt.add_option("--beta", action="store", type="float",
                       dest="beta", default=self.defaults["alpha"],
                       help="Set the artificial viscosity parameter beta")

        opt.add_option("--gamma", action="store", type="float",
                       dest="gamma", default=self.defaults["gamma"],
                       help="Set the ratio of specific heats gamma")        

        opt.add_option("--xsph-eps", action="store", type="float",
                       dest="xsph_eps", default=self.defaults.get("xsph_eps"),
                       help="Constant for XSPH")

        return opt
    
    def setup_solver(self, options=None):

        options = options or self.defaults

        gamma = options.get("gamma")
        alpha = options.get("alpha")
        beta = options.get("beta")
        hks = options.get("hks")

        xsph_eps = options.get("xsph_eps")

        vel_updates=["u","v","w"][:self.dim]
        pos_updates=["x","y","z"][:self.dim]

        ###################################################################
        # Add the operations
        ###################################################################        
        
        # Summation density
        self.add_operation(SPHOperation(

            sph.SPHRho.withargs(hks=hks),
            on_types=[Fluids], from_types=[Fluids, base.Boundary], 
            updates=['rho'], id = 'density')

                           )

        # Equation of state
        self.add_operation(SPHOperation(

            sph.IdealGasEquation.withargs(gamma=gamma),
            on_types = [Fluids],
            updates=['p', 'cs'],
            id='eos')

                           )

        # Momentum equation
        self.add_operation(SPHIntegration(

            sph.MomentumEquation.withargs(alpha=alpha, beta=beta, hks=hks),
            on_types=[Fluids], from_types=[Fluids, base.Boundary], 
            updates=vel_updates,
            id='mom')

                           )

        # XSPH correction : defaults to eps = 0
        self.add_operation(SPHIntegration(

            sph.XSPHCorrection.withargs(eps=xsph_eps),
            on_types=[base.Fluid,], from_types=[base.Boundary, base.Fluid],
            updates=vel_updates,
            id="xsph")

                           )

        # Energy Equation
        self.add_operation(SPHIntegration(

            sph.EnergyEquation.withargs(hks=hks),
            from_types=[Fluids, base.Boundary],
            on_types=[Fluids], updates=['e'],
            id='enr')

                           )

        # Position Step
        self.add_operation(SPHIntegration(

            sph.PositionStepping.withargs(),
            on_types=[Fluids,],
            updates=pos_updates,
            id="step")

                           )

############################################################################
# `ADKEShockTubeSolver` class
############################################################################
class ADKEShockTubeSolver(Solver):

    def __init__(self, dim, integrator_type, h0, eps, k, g1, g2, alpha, beta,
                 gamma=1.4, xsph_eps=0,
                 kernel=base.CubicSplineKernel, hks=True):

        # solver dimension
        self.dim = dim

        # Hernquist and Katz normalization
        self.hks = hks

        # the SPH kernel to use
        self.kernel = kernel(dim)

        self.defaults = dict(alpha=alpha,
                             beta=beta,
                             gamma=gamma,
                             adke_eps=eps,
                             adke_k=k,
                             adke_h0=h0,
                             g1=g1,
                             g2=g2,
                             xsph_eps=xsph_eps)

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

        opt.add_option("--gamma", action="store", type="float",
                       dest="gamma", default=self.defaults["gamma"],
                       help="Set the ratio of specific heats gamma")

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

        opt.add_option("--xsph-eps", action="store", type="float",
                       dest="xsph_eps", default=self.defaults.get("xsph_eps"),
                       help="Constant for XSPH")

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

        gamma = options.get("gamma")

        xsph_eps = options.get("xsph_eps")

        vel_updates=["u","v","w"][:self.dim]
        pos_updates=["x","y","z"][:self.dim]

        ###################################################################
        # Add the operations
        ###################################################################

        # reset the smoothing length to h0
        self.add_operation(SPHOperation(

            sph.SetSmoothingLength.withargs(h0=h0),
            on_types=[base.Fluid,],
            updates=["h"],
            id="setsmoothing")

                           )

        # pilot rho estimate
        self.add_operation(SPHOperation(

            sph.ADKEPilotRho.withargs(h0=h0),
            on_types=[base.Fluid,], from_types=[base.Fluid, base.Boundary],
            updates=['rhop'], id='adke_rho'),

                        )

        # smoothing length update
        self.add_operation(SPHOperation(
            
            sph.ADKESmoothingUpdate.withargs(h0=h0, k=k, eps=eps, hks=hks),
            on_types=[base.Fluid,],
            updates=['h'], id='adke'),
                        
                        )

        # summation density
        self.add_operation(SPHOperation(
            
            sph.SPHRho.withargs(hks=hks),
            on_types=[base.Fluid,], from_types=[base.Fluid,base.Boundary], 
            updates=['rho'], id = 'density')
                        
                        )

        # ideal gas equation
        self.add_operation(SPHOperation(
            
            sph.IdealGasEquation.withargs(gamma=gamma),
            on_types = [base.Fluid,], updates=['p', 'cs'], id='eos')
                        
                        )

        # velocity divergence
        self.add_operation(SPHOperation(
            
            sph.VelocityDivergence.withargs(hks=hks),
            on_types=[base.Fluid], from_types=[base.Fluid, base.Boundary],
            updates=['div'], id='vdivergence'),
                        
                    )

        # conduction coefficient update
        self.add_operation(SPHOperation(
            
            sph.ADKEConductionCoeffUpdate.withargs(g1=g1, g2=g2),
            on_types=[base.Fluid],
            updates=['q'], id='qcoeff'),
                        
                        )

        # momentum equation
        self.add_operation(SPHIntegration(
    
            sph.MomentumEquation.withargs(alpha=alpha, beta=beta, hks=hks),
            from_types=[base.Fluid, base.Boundary], on_types=[base.Fluid], 
            updates=vel_updates, id='mom')
                        
                        )

        # XSPH correction : defaults to eps = 0
        self.add_operation(SPHIntegration(

            sph.XSPHCorrection.withargs(eps=xsph_eps, hks=hks),
            on_types=[base.Fluid,], from_types=[base.Boundary, base.Fluid],
            updates=vel_updates,
            id="xsph")

                           )

        # energy equation
        self.add_operation(SPHIntegration(
            
            sph.EnergyEquation.withargs(hks=hks,alpha=alpha, beta=beta,
                                        gamma=gamma),
            on_types=[base.Fluid], from_types=[base.Fluid, base.Boundary],
            updates=['e'],
            id='enr')
                        
                        )

        # artificial heat 
        self.add_operation(SPHIntegration(
            
            sph.ArtificialHeat.withargs(eta=0.1, hks=hks),
            on_types=[base.Fluid], from_types=[base.Fluid,base.Boundary],
            updates=['e'],
            id='aheat'),
                        
                        )
        
        # position step
        self.add_operation(SPHIntegration(

            sph.PositionStepping.withargs(),
            on_types=[Fluids,],
            updates=pos_updates,
            id="step")

                           )

############################################################################
# `MonaghanShockTubeSolver` class
############################################################################
class MonaghanShockTubeSolver(Solver):

    def __init__(self, dim, integrator_type, h0, eps, k,
                 beta=1.0, K=1.0, f=0.5, gamma=1.4,
                 xsph_eps=0.0, summation_density=True,
                 kernel=base.CubicSplineKernel, hks=True):

        # set the solver dimension
        self.dim = dim

        # Hernquist and Katz normalization
        self.hks = hks

        # the SPH kernel to use
        self.kernel = kernel(dim)

        # set the defaults
        self.defaults = dict(gamma=gamma,
                             adke_eps=eps, adke_k=k, adke_h0=h0,
                             beta=beta, K=K, f=f,
                             xsph_eps=xsph_eps,
                             summation_density=summation_density)

        # base class constructor
        Solver.__init__(self, dim, integrator_type)

    def get_options(self, opt_parser):

        opt = OptionGroup(opt_parser, "MonaghanShockTubeSolver options")

        opt.add_option("--sd", action="store_true",
                       dest="summation_density",
                       default=self.defaults["summation_density"],
                       help="Use summation density for the density equation")

        opt.add_option("--gamma", action="store", type="float",
                       dest="gamma", default=self.defaults["gamma"],
                       help="Set the ratio of specific heats gamma")

        opt.add_option("--adke-eps", action="store", type="float",
                       dest="adke_eps", default=self.defaults.get("adke_eps"),
                       help="Sensitivity parameter eps for the ADKE pocedure")

        opt.add_option("--adke-k", action="store", type="float",
                       dest="adke_k", default=self.defaults.get("adke_k"),
                       help="Scaling parameter k for the ADKE pocedure")

        opt.add_option("--adke-h0", action="store", type="float",
                       dest="adke_h0", default=self.defaults.get("adke_h0"),
                       help="Initial smoothing length h0 for the ADKE pocedure")

        opt.add_option("--beta", action="store", type="float",
                       dest="beta", default=self.defaults["beta"],
                       help="Constant 'beta' for the signal viscosity")

        opt.add_option("--f", action="store", type="float",
                       dest="f", default=self.defaults.get("beta"),
                       help="Constant 'f' for the signal viscosity")

        opt.add_option("--K", action="store", type="float",
                       dest="K", default=self.defaults.get("K"),
                       help="Constant 'K' for the signal viscosity")

        opt.add_option("--xsph-eps", action="store", type="float",
                       dest="xsph_eps", default=self.defaults.get("xsph_eps"),
                       help="Constant for XSPH")

        return opt 

    def setup_solver(self, options=None):

        options = options or self.defaults

        hks = options.get("hks")

        # ADKE parameters
        h0 = options.get("adke_h0")
        eps = options.get("adke_eps")
        k = options.get("adke_k")

        # Artificial viscosity parameters
        beta = options.get("beta")
        K = options.get("K")
        f = options.get("f")

        gamma = options.get("gamma")

        # XSPH eps
        xsph_eps = options.get("xsph_eps")

        # summation density
        sd = options.get("summation_density")

        vel_updates=["u","v","w"][:self.dim]
        pos_updates=["x","y","z"][:self.dim]

        ###################################################################
        # Add the operations
        ###################################################################

        # reset the smoothing length to h0
        self.add_operation(SPHOperation(

            sph.SetSmoothingLength.withargs(h0=h0),
            on_types=[base.Fluid,],
            updates=["h"],
            id="setsmoothing")

                           )        

        # pilot rho estimate
        self.add_operation(SPHOperation(

            sph.ADKEPilotRho.withargs(h0=h0),
            on_types=[Fluids,], from_types=[Fluids,Boundary],
            updates=['rhop'], id='adke_rho'),

                        )

        # smoothing length update
        self.add_operation(SPHOperation(
            
            sph.ADKESmoothingUpdate.withargs(h0=h0, k=k, eps=eps, hks=hks),
            on_types=[Fluids,],
            updates=['h'], id='adke'),
                        
                        )
        # summation density if requested
        if sd:
            self.add_operation(SPHOperation(

                sph.SPHRho.withargs(hks=hks),
                on_types=[base.Fluid,], from_types=[base.Fluid, base.Boundary],
                updates=["rho"],
                id="summation_density")

                               )

        # ideal gas eos
        self.add_operation(SPHOperation(
    
            sph.IdealGasEquation.withargs(gamma=gamma),
            on_types = [base.Fluid],
            updates=['p', 'cs'],
            id='eos')
                        
                        )
        # density rate if not summation density
        if not sd:
            self.add_operation(SPHIntegration(

                sph.SPHDensityRate.withargs(hks=hks),
                on_types=[base.Fluid,], from_types=[base.Boundary, base.Fluid],
                updates=["rho"],
                id="densityrate")

                           )

        # momentum equation pressure gradient
        self.add_operation(SPHIntegration(

            sph.SPHPressureGradient.withargs(hks=hks),
            on_types=[base.Fluid,], from_types=[base.Boundary, base.Fluid],
            updates=vel_updates,
            id="pgrad")

                           )

        # momentum equation artificial viscosity
        self.add_operation(SPHIntegration(

            sph.MomentumEquationSignalBasedViscosity.withargs(beta=beta,
                                                              K=K,
                                                              hks=hks),
            on_types=[base.Fluid,], from_types=[base.Boundary, base.Fluid],
            updates=vel_updates,
            id="visc")

                           )

        # XSPH correction : defaults to eps = 0
        self.add_operation(SPHIntegration(

            sph.XSPHCorrection.withargs(eps=xsph_eps, hks=hks),
            on_types=[base.Fluid,], from_types=[base.Boundary, base.Fluid],
            updates=vel_updates,
            id="xsph")

                           )

        # energy equation
        self.add_operation(SPHIntegration(

            sph.EnergyEquationWithSignalBasedViscosity.withargs(beta=beta,
                                                                K=K,f=f,
                                                                hks=hks),
            on_types=[base.Fluid,], from_types=[base.Boundary, base.Fluid],
            updates=["e"],
            id="energy")

                           )

        # position step
        self.add_operation(SPHIntegration(

            sph.PositionStepping.withargs(),
            on_types=[Fluids,],
            updates=pos_updates,
            id="step")

                           )

############################################################################
# `ShockTubeSolver` class
############################################################################
class GSPHShockTubeSolver(Solver):

    def __init__(self, dim, integrator_type=None):

        self.dim = dim

        integrator_type = GSPHIntegrator

        # base class constructor
        Solver.__init__(self, dim, integrator_type)

        self.default_kernel = base.GaussianKernel(dim)

    def get_options(self, opt_parser):
        pass
    
    def setup_solver(self, options=None):

        ###################################################################
        # Add the operations
        ###################################################################

        hks=False

        vel_updates=["u","v","w"][:self.dim]
        pos_updates=["x","y","z"][:self.dim]
        
        # Summation density
        self.add_operation(SPHOperation(

            sph.SPHRho.withargs(hks=hks),
            on_types=[Fluids], from_types=[Fluids, base.Boundary], 
            updates=['rho'], id = 'density')

                           )

        # Equation of state
        self.add_operation(SPHOperation(

            sph.IdealGasEquation.withargs(gamma=1.4),
            on_types = [Fluids],
            updates=['p', 'cs'],
            id='eos')

                           )

        # Momentum equation
        self.add_operation(SPHIntegration(

            sph.GSPHMomentumEquation.withargs(gamma=1.4),
            on_types=[Fluids], from_types=[Fluids, base.Boundary], 
            updates=vel_updates,
            id='mom')

                           )

        # Energy Equation
        self.add_operation(SPHIntegration(

            sph.GSPHEnergyEquation.withargs(hks=hks),
            from_types=[Fluids, base.Boundary],
            on_types=[Fluids], updates=['e'],
            id='enr')

                           )

        # Position Step
        self.add_operation(SPHIntegration(

            sph.PositionStepping.withargs(),
            on_types=[Fluids,],
            updates=pos_updates,
            id="step")

                           )        
