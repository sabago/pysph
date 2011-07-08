""" A simple shock tube solver """

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
    
    def setup_solver(self, options=None):

        kernel = base.CubicSplineKernel(dim=1)

        #create the sph operation objects

        self.add_operation(SPHOperation(

            sph.SPHRho.withargs(),
            from_types=[Fluids], on_types=[Fluids], 
            updates=['rho'], id = 'density', kernel=kernel)

                           )

        self.add_operation(SPHOperation(

            sph.IdealGasEquation.withargs(),
            on_types = [Fluids], updates=['p', 'cs'], id='eos',
            kernel=kernel)

                           )

        self.add_operation(SPHIntegration(

            sph.MomentumEquation.withargs(),
            from_types=[Fluids], on_types=[Fluids], 
            updates=['u'], id='mom', kernel=kernel)

                           )
        
        self.add_operation(SPHIntegration(

            sph.EnergyEquation.withargs(hks=False),
            from_types=[Fluids],
            on_types=[Fluids], updates=['e'], id='enr',
            kernel=kernel)

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

        # base class constructor
        Solver.__init__(self, dim, integrator_type)
    
    def setup_solver(self, options=None):

        hks = self.hks
        kernel = self.kernel

        # ADKE parameters
        h0 = self.h0
        eps = self.adke_eps
        k = self.k

        # Artificial heat parameters
        g1 = self.g1
        g2 = self.g2

        # Artificial viscosity parameters
        alpha = self.alpha
        beta = self.beta

        self.add_operation(SPHOperation(

            sph.ADKEPilotRho.withargs(h0=h0),
            on_types=[Fluids], from_types=[Fluids,Boundary],
            updates=['rhop'], id='adke_rho', kernel=kernel),

                        )

        # smoothing length update
        self.add_operation(SPHOperation(
            
            sph.ADKESmoothingUpdate.withargs(h0=h0, k=k, eps=eps, hks=hks),
            on_types=[Fluids], updates=['h'], id='adke', kernel=kernel),
                        
                        )

        # summation density
        self.add_operation(SPHOperation(
            
            sph.SPHRho.withargs(hks=hks),
            from_types=[Fluids, Boundary], on_types=[Fluids], 
            updates=['rho'], id = 'density', kernel=kernel)
                        
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
        
        
        
        
    
        
        
