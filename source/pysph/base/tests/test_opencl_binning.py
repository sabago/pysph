""" A test script whech checks the OpenCL binning with respect to the default
Cython binning.

The particle state at each iteration in the shock tube problem is
passed on to the OpenCL domain manager for binning and the result is
compared with the Cyhton binning. 


"""
import logging

import pysph.base.api as base
import pysph.solver.api as solver
from pysph.base.kernels import CubicSplineKernel

from pysph.base.linked_list_functions import cell_neighbors

if not solver.HAS_CL:
    try:
        import nose.plugins.skip as skip
        reason = "PyOpenCL not installed"
        raise skip.SkipTest(reason)
    except ImportError:
        pass

import unittest

CLDomain = base.DomainManagerType
CLLocator = base.OpenCLNeighborLocatorType
Locator = base.NeighborLocatorType

def domain_update(domain_manager):
    domain_manager.find_bounds()
    domain_manager.init_linked_list()
    domain_manager.cl_update()

from pysph.base.cell import py_find_cell_id
def get_cellid(idx, cell_size):
    pnt = base.Point(pa.x[idx])
    return py_find_cell_id(pnt, cell_size).x    

class TestShockTube(unittest.TestCase):

    def test(self):

        # Cython setup
        pa = solver.shock_tube_solver.standard_shock_tube_data(
            name="test", cl_precision="double", type=0)

        s = solver.ShockTubeSolver(dim=1,
                                   integrator_type=solver.EulerIntegrator)

        particles = base.Particles(arrays=[pa,])

        s.setup(particles)
        s.set_final_time(0.15)
        s.set_time_step(3e-4)

        loc = particles.get_neighbor_particle_locator(pa, pa, 2.0)
        integrator = s.integrator
        cell_manager = particles.cell_manager

        # OpenCL setup
        ctx = solver.create_some_context()
        domain_manager = base.LinkedListManager(arrays=[pa,], context=ctx)
        assert ( domain_manager.with_cl == True )
        
        domain_update(domain_manager)

        t = 0.0
        dt = 3e-4
        tf = 0.15
        np = 400

        assert (domain_manager.cell_size == cell_manager.cell_size)
        cell_size = domain_manager.cell_size
        
        while t < tf:
            
            particles.update()
            integrator.integrate(dt)

            cell_manager.update()

            domain_update(domain_manager)
            domain_manager.enqueue_copy()

            head = domain_manager.head["test"]
            next = domain_manager.next["test"]
            cellids = domain_manager.cellids["test"]

            for i in range(np):
                index = loc.get_cellid_dest(i)
                
                cell = cell_manager.cells_dict[index]

                cy_nbrs = cell.index_lists[0].get_npy_array()
                cy_nbrs.sort()
        
                cl_nbrs = cell_neighbors( cellids[i], head, next )
                cl_nbrs.sort()

                nclnbrs = len(cl_nbrs)
                ncynbrs = len(cy_nbrs)

                assert ( nclnbrs == ncynbrs )

                for j in range(ncynbrs):
                    assert ( cl_nbrs[j] ==  cy_nbrs[j] )

            t += dt


if __name__ == "__main__":
    unittest.main()
