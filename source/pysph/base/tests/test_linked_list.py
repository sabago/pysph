""" Test the linked list functions """

import numpy
import numpy.random as random

import unittest

# PySPH imports
import pysph.base.api as base
from pysph.solver.api import get_real, HAS_CL

# Cython functions for the neighbor list
import pysph.base.linked_list_functions as ll

if not HAS_CL:
    try:
        import nose.plugins.skip as skip
        reason = "PyOpenCL not installed"
        raise skip.SkipTest(reason)
    except ImportError:
        pass

class SinglePrecisionLinkedListManagerTestCase(unittest.TestCase):

    def setUp(self):
        """ The setup consists of points randomly distributed in a
        cube [-1,1] X [-1.1] X [1-,1]

        The smoothing length for the points is proportional to the
        number of particles.

        """

        self.cl_precision = cl_precision = "single"
        self.np = np = 10001

        self.x = x = random.random(np) * 2.0 - 1.0
        self.y = y = random.random(np) * 2.0 - 1.0
        self.z = z = random.random(np) * 2.0 - 1.0

        vol_per_particle = numpy.power(8.0/np, 1.0/3.0)

        self.h = h = numpy.ones_like(x) * vol_per_particle

        self.cy_pa = base.get_particle_array(name='test', x=x, y=y, z=z, h=h)

        self.cl_pa = base.get_particle_array(
            name="test", cl_precision=self.cl_precision,
            x=x, y=y, z=z, h=h)

        # the scale factor for the cell sizes
        self.kernel_scale_factor = kernel_scale_factor = 2.0

        self.cl_manager = base.LinkedListManager(
            arrays=[self.cl_pa,],
            kernel_scale_factor=kernel_scale_factor)

        self.cy_manager = base.LinkedListManager(
            arrays=[self.cy_pa,], with_cl=False,
            kernel_scale_factor=kernel_scale_factor)

        self.cy_manager.cl_precision = "single"

        # construct the neighbor locator
        self.loc = base.LinkedListSPHNeighborLocator(
            manager=self.cy_manager,
            source=self.cy_pa, dest=self.cy_pa)

    def test_neighbor_locator_construct(self):
        """ Test the constructor for the LinkedListSPHNeighborLocator.
        """
        loc = self.loc

        self.assertEqual( loc.cache, False )
        self.assertEqual( loc.with_cl, False )
        self.assertAlmostEqual( loc.scale_fac, 2.0, 10 )

        self.assertEqual( loc.particle_cache, [] )

    def test_constructor(self):
        """ Test the constructor for the LinkedListManager.
        """

        cl_manager = self.cl_manager
        cy_manager = self.cy_manager

        # check the with_cl flag
        self.assertEqual(cy_manager.with_cl, False)
        self.assertEqual(cl_manager.with_cl, True)

        # get the global simulation bounds for the data
        mx, my, mz = min(self.x), min(self.y), min(self.z)
        Mx, My, Mz = max(self.x), max(self.y), max(self.z)

        # convert to single precision
        mx = get_real(mx, self.cl_precision)
        my = get_real(my, self.cl_precision)
        mz = get_real(mz, self.cl_precision)

        Mx = get_real(Mx, self.cl_precision)
        My = get_real(My, self.cl_precision)
        Mz = get_real(Mz, self.cl_precision)

        Mh = get_real(max(self.h), self.cl_precision)

        # get the cell size for the domain manager. Remember that we
        # choose (k + 1) as the scaling constant.
        cell_size = (self.kernel_scale_factor + 1) * Mh

        # convert the cell sizes to single precision
        cy_cell_size = get_real(cell_size, self.cl_precision)
        cl_cell_size = get_real(cell_size, self.cl_precision)

        # check the simulation bounds.
        self.assertAlmostEqual(cl_manager.mx, mx, 8)
        self.assertAlmostEqual(cl_manager.my, my, 8)
        self.assertAlmostEqual(cl_manager.mz, mz, 8)

        self.assertAlmostEqual(cl_manager.Mx, Mx, 8)
        self.assertAlmostEqual(cl_manager.My, My, 8)
        self.assertAlmostEqual(cl_manager.Mz, Mz, 8)        

        self.assertAlmostEqual(cl_manager.Mh, Mh, 8)

        # check the cell sizes for the Cython and OpenCL managers
        self.assertAlmostEqual(cl_manager.cell_size, cl_cell_size, 8)
        self.assertAlmostEqual(cy_manager.cell_size, cy_cell_size, 8)

    def test_bin_particles(self):
        """ Test the Cyhton and OpenCL binning.

        The particles are binned with respect to the two managers and
        the cell indices for the binnings are compared. What this test
        effectively checks is that the two operations are equivalent.
        
        """
        cy_manager = self.cy_manager
        cl_manager = self.cl_manager

        # update the bin structure
        cy_manager.update()
        cl_manager.update()

        # read the OpenCL data from the device buffer
        cl_manager.enqueue_copy()

        # get the binning data for each manager
        cy_bins = cy_manager.cellids[self.cy_pa.name]
        cl_bins = cl_manager.cellids[self.cl_pa.name]

        cy_ix = cy_manager.ix[self.cy_pa.name]
        cy_iy = cy_manager.iy[self.cy_pa.name]
        cy_iz = cy_manager.iz[self.cy_pa.name]

        cl_ix = cy_manager.ix[self.cl_pa.name]
        cl_iy = cy_manager.iy[self.cl_pa.name]
        cl_iz = cy_manager.iz[self.cl_pa.name]

        # a bin should be created for each particle
        self.assertEqual( len(cy_bins), self.np )
        self.assertEqual( len(cl_bins), self.np )

        # check the cellid (flattened and unflattened) for each particle
        for i in range(self.np):
            self.assertEqual( cy_bins[i], cl_bins[i] )

            self.assertEqual( cy_ix[i], cl_ix[i] )
            self.assertEqual( cy_iy[i], cl_iy[i] )
            self.assertEqual( cy_iz[i], cl_iz[i] )

    def test_construct_neighbor_list(self):
        """ Test construction of the neighbor lists.

        The two domain managers are used to independently bin and
        construct the neighbor list for each particle. The neighbors
        are then compared. This test establshes that OpenCL and
        Cython produce the same neighbor lists, and thus are equivalent.

        """

        name = self.cy_pa.name
        
        cy_manager = self.cy_manager
        cl_manager = self.cl_manager

        # update the structure 
        cy_manager.update()
        cl_manager.update()

        # read the buffer contents for the OpenCL manager
        cl_manager.enqueue_copy()

        # the number of cells should be the same
        ncells = len(cy_manager.head[name])
        self.assertEqual( len(cy_manager.head[name]), len(cl_manager.head[name]) )
        
        # check the neighbor lists
        cy_bins = cy_manager.cellids[name]
        cy_head = cy_manager.head[name]
        cy_next = cy_manager.next[name]
        
        cl_bins = cl_manager.cellids[name]
        cl_head = cl_manager.head[name]
        cl_next = cl_manager.next[name]

        for i in range(self.np):

            cy_nbrs = ll.cell_neighbors( cy_bins[i], cy_head, cy_next )
            cl_nbrs = ll.cell_neighbors( cl_bins[i], cl_head, cl_next )

            cy_nbrs.sort()
            cl_nbrs.sort()

            # the number of neighbors should be the same
            nnbrs = len(cy_nbrs)
            self.assertEqual( len(cl_nbrs), nnbrs )

            # the sorted list of neighbors should be the same
            for j in range( nnbrs ):
                self.assertEqual( cl_nbrs[j], cy_nbrs[j] )

    def test_neighbor_locator(self):
        """Test the neighbors returned by the OpenCL locator.

        The neighbors for each particle returned by a locator based on
        the Cython domain manager are compared with the brute force
        neighbors. This test establishes that the true neighbors are
        returned by the Cython domain manager.

        Since the OpenCL and Cython generate equivalent neighbors, we
        conclude that the OpenCL neighbors are also correct.

        """

        loc = self.loc

        # update the data structure
        self.cy_manager.update()

        # get the co-ordinate data in single precision
        x = self.x.astype(numpy.float32)
        y = self.y.astype(numpy.float32)
        z = self.z.astype(numpy.float32)
        h = self.h.astype(numpy.float32)
        
        for i in range(self.np):

            xi, yi, zi = x[i], y[i], z[i]

            # the search radius is (k * hi)
            radius = loc.scale_fac * h[i]
            brute_nbrs = ll.brute_force_neighbors(xi, yi, zi,
                                                  self.np,
                                                  radius,
                                                  x, y, z)

            # get the neighbors with Cython
            nbrs = base.LongArray()
            loc.get_nearest_particles(i, nbrs)
            loc_nbrs = nbrs.get_npy_array()
            loc_nbrs.sort()

            # the number of neighbors should be the same
            nnbrs = len(loc_nbrs)
            self.assertEqual(len(loc_nbrs), len(brute_nbrs))

            # each neighbor in turn should be the same when sorted
            for j in range(nnbrs):
                self.assertEqual( loc_nbrs[j], brute_nbrs[j] )

if __name__ == '__main__':
    unittest.main()
