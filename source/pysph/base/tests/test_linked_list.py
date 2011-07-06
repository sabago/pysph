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

        self.cell_size = 2 * vol_per_particle

        self.h = h = numpy.ones_like(x) * vol_per_particle

        self.cy_pa = base.get_particle_array(name='test', x=x, y=y, z=z, h=h)

        self.cl_pa = base.get_particle_array(
            name="test", cl_precision=self.cl_precision,
            x=x, y=y, z=z, h=h)

        self.cl_manager = base.LinkedListManager([self.cl_pa,])

        self.cy_manager = base.LinkedListManager([self.cy_pa,], with_cl=False)
        self.cy_manager.cl_precision = "single"

        # construct the neighbor locator
        self.loc = base.LinkedListSPHNeighborLocator(
            manager=self.cy_manager,
            source=self.cy_pa, dest=self.cy_pa)

    def test_neighbor_locator_construct(self):

        loc = self.loc

        self.assertEqual( loc.cache, False )
        self.assertEqual( loc.with_cl, False )
        self.assertAlmostEqual( loc.scale_fac, 2.0, 10 )

        self.assertEqual( loc.particle_cache, [] )

        self.assertEqual( loc.is_dirty, True )


    def test_constructor(self):

        mx, my, mz = min(self.x), min(self.y), min(self.z)
        Mx, My, Mz = max(self.x), max(self.y), max(self.z)

        cl_manager = self.cl_manager
        cy_manager = self.cy_manager

        mx = get_real(mx, self.cl_precision)
        my = get_real(my, self.cl_precision)
        mz = get_real(mz, self.cl_precision)

        Mx = get_real(Mx, self.cl_precision)
        My = get_real(My, self.cl_precision)
        Mz = get_real(Mz, self.cl_precision)

        Mh = get_real(max(self.h), self.cl_precision)

        cy_cell_size = self.cell_size
        cl_cell_size = get_real(self.cell_size, self.cl_precision)

        # check the simulation bounds and cell size
        
        self.assertAlmostEqual(cl_manager.mx, mx, 10)
        self.assertAlmostEqual(cl_manager.my, my, 10)
        self.assertAlmostEqual(cl_manager.mz, mz, 10)

        self.assertAlmostEqual(cl_manager.Mx, Mx, 10)
        self.assertAlmostEqual(cl_manager.My, My, 10)
        self.assertAlmostEqual(cl_manager.Mz, Mz, 10)        

        self.assertAlmostEqual(cl_manager.Mh, Mh, 10)

        self.assertAlmostEqual(cl_manager.cell_size, cl_cell_size, 10)
        self.assertAlmostEqual(cy_manager.cell_size, cy_cell_size, 10)

        # check the with_cl flag

        self.assertEqual(cy_manager.with_cl, False)
        self.assertEqual(cl_manager.with_cl, True)

        # check the dirty flag

        self.assertEqual(cl_manager.is_dirty, True)
        self.assertEqual(cy_manager.is_dirty, True)

        # check that the particle array is dirty

        self.assertEqual(self.cy_pa.is_dirty, True)
        self.assertEqual(self.cl_pa.is_dirty, True)

        
    def test_bin_particles(self):
        """ Bin the particles using Cython and OpenCL and compare the results """

        cy_manager = self.cy_manager
        cl_manager = self.cl_manager

        # update the bin structure
        cy_manager.update()
        cl_manager.update()

        # the particle arrays should not be dirty now

        self.assertEqual(self.cy_pa.is_dirty, False)
        self.assertEqual(self.cl_pa.is_dirty, False)

        # check the particle cell indices
        cl_manager.enqueue_copy()

        cy_bins = cy_manager.cellids[self.cy_pa.name]
        cl_bins = cl_manager.cellids[self.cl_pa.name]

        cy_ix = cy_manager.ix[self.cy_pa.name]
        cy_iy = cy_manager.iy[self.cy_pa.name]
        cy_iz = cy_manager.iz[self.cy_pa.name]

        cl_ix = cy_manager.ix[self.cl_pa.name]
        cl_iy = cy_manager.iy[self.cl_pa.name]
        cl_iz = cy_manager.iz[self.cl_pa.name]

        self.assertEqual( len(cy_bins), self.np )
        self.assertEqual( len(cl_bins), self.np )

        for i in range(self.np):
            self.assertEqual( cy_bins[i], cl_bins[i] )

            self.assertEqual( cy_ix[i], cl_ix[i] )
            self.assertEqual( cy_iy[i], cl_iy[i] )
            self.assertEqual( cy_iz[i], cl_iz[i] )

    def test_construct_neighbor_list(self):

        name = self.cy_pa.name
        
        cy_manager = self.cy_manager
        cl_manager = self.cl_manager

        # update the structure 
        cy_manager.update()
        cl_manager.update()

        # read the buffer contents
        cl_manager.enqueue_copy()

        # the number of cells should be the same

        self.assertEqual( len(cy_manager.head[name]), len(cl_manager.head[name]) )

        ncells = len(cy_manager.head[name])
        
        # check the neighbor list by getting cells within a neighbor
        
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

            nnbrs = len(cy_nbrs)

            self.assertEqual( len(cl_nbrs), nnbrs )

            for j in range( nnbrs ):
                self.assertEqual( cl_nbrs[j], cy_nbrs[j] )

    def test_neighbor_locator(self):

        loc = self.loc

        # perform an update for the locator
        loc.update_status()

        self.assertEqual( loc.is_dirty, True )
        self.assertEqual( loc.manager.is_dirty, True)

        loc.update()

        self.assertEqual( loc.manager.is_dirty, False )
        self.assertEqual( loc.is_dirty, False )
        self.assertEqual( self.cy_pa.is_dirty, False )

        x = self.x.astype(numpy.float32)
        y = self.y.astype(numpy.float32)
        z = self.z.astype(numpy.float32)
        
        cell_size = self.cell_size

        for i in range(self.np):

            nbrs = base.LongArray()

            xi, yi, zi = x[i], y[i], z[i]

            brute_nbrs = ll.brute_force_neighbors(xi, yi, zi,
                                                  self.np,
                                                  cell_size,
                                                  x, y, z)

            loc.get_nearest_particles(i, nbrs)
            loc_nbrs = nbrs.get_npy_array()
            loc_nbrs.sort()

            nnbrs = len(loc_nbrs)

            self.assertEqual(len(loc_nbrs), len(brute_nbrs))

            for j in range(nnbrs):
                self.assertEqual( loc_nbrs[j], brute_nbrs[j] )

if __name__ == '__main__':
    unittest.main()
