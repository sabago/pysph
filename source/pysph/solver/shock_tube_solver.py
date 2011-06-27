""" A simple shock tube solver """

import numpy

import pysph.base.api as base
import pysph.sph.api as sph

from solver import Solver
from sph_equation import SPHOperation, SPHIntegration

Fluids = base.ParticleType.Fluid
Solids = base.ParticleType.Solid

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
    
    def setup_solver(self):

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
    
        
        
