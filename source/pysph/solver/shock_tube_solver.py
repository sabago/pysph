""" A simple shock tube solver """

import numpy

import pysph.base.api as base
import pysph.sph.api as sph

#from pysph.base.particle_types import ParticleType
#from pysph.base.particle_array import get_particle_array

from solver import Solver
from sph_equation import SPHSummation, SPHAssignment, SPHSummationODE, \
    SPHSimpleODE

Fluids = base.ParticleType.Fluid
Solids = base.ParticleType.Solid

def standard_shock_tube_data(name="", type=0):
    """ Standard 400 particles shock tube problem """
    
    dxl = 0.001875
    dxr = dxl*4
    
    x = numpy.ones(400, float)
    x[:320] = numpy.arange(-0.6, -dxl+1e-4, dxl)
    x[320:] = numpy.arange(dxr, 0.6+1e-4, dxr)

    m = numpy.ones_like(x)*dxl
    h = numpy.ones_like(x)*2*dxr

    rho = numpy.ones_like(x)
    rho[320:] = 0.25
    
    u = numpy.zeros_like(x)
    
    e = numpy.ones_like(x)
    e[:320] = 2.5
    e[320:] = 1.795

    p = 0.4*rho*e
    
    return base.get_particle_array(name=name,x=x,m=m,h=h,rho=rho,p=p,e=e,
                                   type=type)

class ShockTubeSolver(Solver):
    
    def setup_solver(self):

        #create the sph operation objects

        self.add_operation(SPHSummation(sph.SPHRho(), 
                                        from_types=[Fluids],
                                        on_types=[Fluids], 
                                        updates=['rho'], 
                                        id = 'density')
                           )

        self.add_operation(SPHAssignment(sph.IdealGasEquation(),
                                         on_types = [Fluids],
                                         updates=['p'],
                                         id='eos')
                           )

        self.add_operation(SPHSummationODE(sph.MomentumEquation(),
                                           from_types=[Fluids],
                                           on_types=[Fluids], 
                                           updates=['u'], 
                                           id='mom')
                           )

        self.add_operation(SPHSummationODE(sph.EnergyEquation(),
                                           from_types=[Fluids],
                                           on_types=[Fluids], 
                                           updates=['e'], id='enr')
                           )

        self.add_operation(SPHSimpleODE(sph.PositionStepping(),
                                        on_types=[Fluids], 
                                        updates=['x'], id='step')
                           )

#############################################################################
    
        
        
