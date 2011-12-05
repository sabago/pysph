""" A test script whech checks the OpenCL binning with respect to the default
Cython binning.

The particle state at each iteration in the shock tube problem is
passed on to the OpenCL domain manager for binning and the result is
compared with the Cyhton binning. 


"""
import logging

import pysph.base.api as base
import pysph.solver.api as solver

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

from pysph.base.nnps_util import ll_cell_neighbors
from pysph.base.kernels import CubicSplineKernel
from pysph.base.cell import py_find_cell_id

class TestShockTube(unittest.TestCase):
    """This is a 'dynamic' test for the OpenCL binning strategy using
    linked lists.

    The standard shock tube problem is simulated and at each time
    step, we ensure that the cell structures due to Cython and OpenCL
    are the same.

    """
    def test(self):
        """Dynamic OpenCL binning test.
        """
        # Create a ParticleArray with double precision
        pa = solver.shock_tube_solver.standard_shock_tube_data(
            name="test", cl_precision="double", type=base.Fluid)

        # get the coordinate array
        x = pa.get('x')

        # set the scale factor and cell size. Remember that in the
        # OpenCL DomainManager, we want the bin size to be (k+1)*maxH
        scale_fac = 2.0
        h0 = pa.h[0]
        cell_size = (scale_fac + 1) * h0

        # create a shock tube solver
        s = solver.ShockTubeSolver(dim=1,
                                   integrator_type=solver.EulerIntegrator)

        # create a Particles instance with a fixed cell size provided. 
        particles = base.Particles(arrays=[pa,],
                                   min_cell_size=cell_size,
                                   max_cell_size=cell_size)

        s.setup(particles)
        s.set_final_time(0.15)
        s.set_time_step(3e-4)

        integrator = s.integrator
        cell_manager = particles.cell_manager

        # Setup the OpenCL domain manager
        ctx = solver.create_some_context()
        domain_manager = base.LinkedListManager(arrays=[pa,], context=ctx)
        assert ( domain_manager.with_cl == True )

        # bin the particles on the OpenCL device
        domain_manager.update()

        # the cell sizes for Cython and OpenCL should be the same
        assert (domain_manager.cell_size == cell_manager.cell_size)
        cell_size = domain_manager.cell_size
                
        t = 0.0
        tf = 0.15
        dt = 3e-4
        np = 400
        while t < tf:

            # update the particles
            particles.update()

            # integrate
            integrator.integrate(dt)

            # call the cell manager's update
            cell_manager.update()

            # now bin the updated data using OpenCL
            domain_manager.update()
            domain_manager.enqueue_copy()

            head = domain_manager.head["test"]
            next = domain_manager.Next["test"]
            cellids = domain_manager.cellids["test"]

            # test the bin structure for each particle
            for i in range(np):

                # find the index of the particle
                pnt = base.Point(x[i])
                index = py_find_cell_id(pnt, cell_size)

                # get the particles in the cell
                cell = cell_manager.cells_dict[index]
                cy_nbrs = cell.index_lists[0].get_npy_array()
                cy_nbrs.sort()

                # get the particle's index with OpenCL
                cl_index = cellids[i]

                # get the particles in the the cell with OpenCL
                cl_nbrs = ll_cell_neighbors( cellids[i], head, next )
                cl_nbrs.sort()

                # the lenght of the neighbors should be the same
                nclnbrs = len(cl_nbrs)
                ncynbrs = len(cy_nbrs)
                assert ( nclnbrs == ncynbrs )

                # test each neighbor
                for j in range(ncynbrs):
                    assert ( cl_nbrs[j] ==  cy_nbrs[j] )

            t += dt

if __name__ == "__main__":
    unittest.main()
