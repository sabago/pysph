"""Tests for the different neighbor locators"""

import numpy
import numpy.random as random

import unittest

# PySPH imports
import pysph.base.api as base
import pysph.solver.cl_utils as clu

import pysph.base.nnps_util as nnps

from nose.plugins.attrib import attr

if not clu.HAS_CL:
    try:
        import nose.plugins.skip as skip
        reason = "PyOpenCL not installed"
        raise skip.SkipTest(reason)
    except ImportError:
        pass

##########################################################################    
#`LocatorTestCase` class
##########################################################################
class LocatorTestCase(unittest.TestCase):
    """General setup for all the neighbor locators"""

    def setUp(self):
        """ The setup consists of points randomly distributed in a
        cube [-1,1] X [-1.1] X [1-,1]

        The smoothing length for the points is proportional to the
        number of particles.

        """

        self.cl_precision = cl_precision = "single"
        self.np = np = 1<<14

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

        self._setup()

##########################################################################    
#`LinkedListSPHNeighborLocatorTestCase` class
##########################################################################
class LinkedListSPHNeighborLocatorTestCase(LocatorTestCase):

    def _setup(self):

        # create the linked list managers
        self.cl_manager = base.LinkedListManager(
            arrays=[self.cl_pa,],
            kernel_scale_factor=self.kernel_scale_factor)

        self.cy_manager = base.LinkedListManager(
            arrays=[self.cy_pa,], with_cl=False,
            kernel_scale_factor=self.kernel_scale_factor)

        self.cy_manager.cl_precision = "single"

        # construct the neighbor locator
        self.loc = base.LinkedListSPHNeighborLocator(
            manager=self.cy_manager,
            source=self.cy_pa, dest=self.cy_pa)

    def test_constructor(self):
        """ LinkedListSPHNeighborLocator: test_constructor"""
        loc = self.loc

        self.assertEqual( loc.cache, False )
        self.assertEqual( loc.with_cl, False )
        self.assertAlmostEqual( loc.scale_fac, 2.0, 10 )

        self.assertEqual( loc.particle_cache, [] )        

    def test_binning(self):
        """ LinkedListSPHNeighborLocator: test_binning

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

    def test_udpate(self):
        """ LinkedListSPHNeighborLocator: test_update

        The two domain managers are used to independently bin and
        construct the neighbor list for each particle. The neighbors
        within a cell are then compared. This test establshes that
        OpenCL and Cython produce the same neighbor lists per cell,
        and thus are equivalent.

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
        cy_next = cy_manager.Next[name]
        
        cl_bins = cl_manager.cellids[name]
        cl_head = cl_manager.head[name]
        cl_next = cl_manager.Next[name]

        for i in range(self.np):

            cy_nbrs = nnps.ll_cell_neighbors( cy_bins[i], cy_head, cy_next )
            cl_nbrs = nnps.ll_cell_neighbors( cl_bins[i], cl_head, cl_next )

            cy_nbrs.sort()
            cl_nbrs.sort()

            # the number of neighbors should be the same
            nnbrs = len(cy_nbrs)
            self.assertEqual( len(cl_nbrs), nnbrs )

            # the sorted list of neighbors should be the same
            for j in range( nnbrs ):
                self.assertEqual( cl_nbrs[j], cy_nbrs[j] )
                
    @attr(slow=True)
    def test_neighbor_locator(self):
        """LinkedListSPHNeighborLocator: test_neighbor_locator

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
            brute_nbrs = nnps.brute_force_neighbors(xi, yi, zi,
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


############################################################################
#`RadixSortNeighborLocator` class
############################################################################
class RadixSortNeighborLocator(LocatorTestCase):

    def _setup(self):

        # construct the OpenCL and Cython managers
        self.cy_manager = base.RadixSortManager(
            arrays=[self.cy_pa,],
            kernel_scale_factor=self.kernel_scale_factor,
            with_cl=False)

        # construct the neighbor locator
        self.loc = base.RadixSortNeighborLocator(
            manager=self.cy_manager,
            source=self.cy_pa, dest=self.cy_pa)

    @unittest.skipIf(clu.get_cl_devices()['CPU'] == [], 'No CPU device detected')
    def test_udpate_cpu(self):
        """ RadixSortSPHNeighborLocator: test_update

        The two domain managers are used to independently bin and
        construct the neighbor list for each particle. The neighbors
        within a cell are then compared. This test establshes that
        OpenCL and Cython produce the same neighbor lists per cell,
        and thus are equivalent.

        """

        self.cl_manager = base.RadixSortManager(
            arrays=[self.cl_pa,],
            kernel_scale_factor=self.kernel_scale_factor,
            device='CPU')

        name = self.cy_pa.name
        
        cy_manager = self.cy_manager
        cl_manager = self.cl_manager

        # update the structure 
        cy_manager.update()
        cl_manager.update()

        # the number of cells should be the same
        ncells = len(cy_manager.indices[name])
        self.assertEqual( len(cy_manager.indices[name]), len(cl_manager.indices[name]) )
        
        self.assertEqual( len(cy_manager.cell_counts[name]), len(cl_manager.cell_counts[name]) )

        cy_cellids = cy_manager.cellids[name]
        cl_cellids = cl_manager.cellids[name]

        for i in range(self.np):
            self.assertEqual( cy_cellids[i], cl_cellids[i] )
            self.assertEqual( cy_manager.indices[name][i], cl_manager.indices[name][i] )

        for i in range( len(cy_manager.cell_counts[name]) ):    
            self.assertEqual( cy_manager.cell_counts[name][i], cl_manager.cell_counts[name][i] )

    @unittest.skipIf(clu.get_cl_devices()['GPU'] == [],"No GPU device detected")
    def test_udpate_gpu(self):
        """ RadixSortSPHNeighborLocator: test_update

        The two domain managers are used to independently bin and
        construct the neighbor list for each particle. The neighbors
        within a cell are then compared. This test establshes that
        OpenCL and Cython produce the same neighbor lists per cell,
        and thus are equivalent.

        """
        self.cl_manager = base.RadixSortManager(
            arrays=[self.cl_pa,],
            kernel_scale_factor=self.kernel_scale_factor,
            device='GPU')

        name = self.cy_pa.name
        
        cy_manager = self.cy_manager
        cl_manager = self.cl_manager

        # update the structure 
        cy_manager.update()
        cl_manager.update()

        # the number of cells should be the same
        ncells = len(cy_manager.indices[name])
        self.assertEqual( len(cy_manager.indices[name]), len(cl_manager.indices[name]) )
        
        self.assertEqual( len(cy_manager.cell_counts[name]), len(cl_manager.cell_counts[name]) )

        cy_cellids = cy_manager.cellids[name]
        cl_cellids = cl_manager.cellids[name]

        for i in range(self.np):
            self.assertEqual( cy_cellids[i], cl_cellids[i] )
            self.assertEqual( cy_manager.indices[name][i], cl_manager.indices[name][i] )

        for i in range( len(cy_manager.cell_counts[name]) ):    
            self.assertEqual( cy_manager.cell_counts[name][i], cl_manager.cell_counts[name][i] )

    @attr(slow=True)
    def test_neighbor_locator(self):
        """RadixSortNeighborLocator: test_neighbor_locator

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

        # get the updated data structures
        d_indices = self.cy_manager.indices["test"]
        
        for i in range(self.np):

            xi = x[ d_indices[i] ]
            yi = y[ d_indices[i] ]
            zi = z[ d_indices[i] ]
            
            # the search radius is (k * hi)
            radius = loc.scale_fac * h[i]
            brute_nbrs = nnps.brute_force_neighbors(xi, yi, zi,
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

        
if __name__ == "__main__":
    unittest.main()

