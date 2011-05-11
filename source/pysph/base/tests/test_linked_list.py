""" Test the linked list functions """

import numpy
import numpy.random as random

import unittest

# PySPH imports
import pysph.base.api as base
from pysph.solver.api import get_real

# Cython functions for the neighbor list
import pysph.base.linked_list_functions as ll

class SinglePrecisionLinkedListManagerTestCase(unittest.TestCase):

    def setUp(self):

        self.np = np = 1000
        self.cl_precision = cl_precision = "single"

        self.x = x = random.random(np) * 2.0 - 1.0
        self.y = y = random.random(np) * 2.0 - 1.0
        self.z = z = random.random(np) * 2.0 - 1.0

        x = x.astype(numpy.float32)
        y = y.astype(numpy.float32)
        z = z.astype(numpy.float32)

        vol_per_particle = numpy.power(8.0/np, 1.0/3.0)

        self.cell_size = numpy.float32(2 * vol_per_particle)

        self.h = h = numpy.ones_like(x) * vol_per_particle

        self.pa = pa = base.get_particle_array(name='test', x=x, y=y, z=z, h=h,
                                               cl_precision=cl_precision)

        self.cy_ll_manager = base.LinkedListManager([pa,])
        self.cy_ll_manager.with_cl = False

        self.cl_ll_manager = base.LinkedListManager([pa,])

        # construct the neighbor locator
        self.loc = base.LinkedListSPHNeighborLocator(manager=self.cy_ll_manager,
                                                     source=pa, dest=pa)
        
    def test_constructor(self):

        mx, my, mz = min(self.x), min(self.y), min(self.z)
        Mx, My, Mz = max(self.x), max(self.y), max(self.z)

        cl_llm = self.cl_ll_manager
        cy_llm = self.cy_ll_manager

        mx = get_real(mx, self.cl_precision)
        my = get_real(my, self.cl_precision)
        mz = get_real(mz, self.cl_precision)

        Mx = get_real(Mx, self.cl_precision)
        My = get_real(My, self.cl_precision)
        Mz = get_real(Mz, self.cl_precision)

        Mh = get_real(max(self.h), self.cl_precision)
        cell_size = get_real(self.cell_size, self.cl_precision)
        
        self.assertAlmostEqual(cl_llm.mx, mx, 10)
        self.assertAlmostEqual(cl_llm.my, my, 10)
        self.assertAlmostEqual(cl_llm.mz, mz, 10)

        self.assertAlmostEqual(cl_llm.Mx, Mx, 10)
        self.assertAlmostEqual(cl_llm.My, My, 10)
        self.assertAlmostEqual(cl_llm.Mz, Mz, 10)        

        self.assertAlmostEqual(cl_llm.Mh, Mh, 10)

        self.assertAlmostEqual(cl_llm.cell_size, cell_size, 10)

        self.assertAlmostEqual(cl_llm.cell_size, cell_size, 10)
        self.assertAlmostEqual(cy_llm.cell_size, self.cell_size, 10)

        self.assertEqual(cy_llm.with_cl, False)
        self.assertEqual(cl_llm.with_cl, True)

        self.assertEqual(cl_llm.is_dirty, True)
        self.assertEqual(cy_llm.is_dirty, True)

        self.assertEqual(self.pa.is_dirty, True)

    def test_construct_neighbor_list(self):

        cy_llm = self.cy_ll_manager
        cl_llm = self.cl_ll_manager

        # update the structure
        cy_llm.update()
        cl_llm.update()

        pa_name = self.pa.name
        
        for i in range(self.np):

            cellid = cy_llm.cellids[pa_name][i]
            cython_nbrs = ll.cell_neighbors(cellid,
                                            cy_llm.head[pa_name],
                                            cy_llm.next[pa_name],
                                            )

            # read the buffer contents
            cl_llm.enqueue_copy()
            cellid = cl_llm.cellids[pa_name][i]
            opencl_nbrs = ll.cell_neighbors(cellid,
                                            cl_llm.head[pa_name],
                                            cl_llm.next[pa_name],
                                            )

            cython_nbrs.sort()
            opencl_nbrs.sort()

            nnbrs = len(cython_nbrs)
            self.assertEqual( len(cython_nbrs), len(opencl_nbrs) )

            for j in range(nnbrs):
                self.assertEqual( cython_nbrs[j], opencl_nbrs[j] )

        self.assertEqual(self.pa.is_dirty, False)

    def test_neighbor_locator_construct(self):

        loc = self.loc

        self.assertEqual( loc.cache, False )
        self.assertEqual( loc.with_cl, False )
        self.assertAlmostEqual( loc.scale_fac, 2.0, 10 )

        self.assertEqual( loc.particle_cache, [] )

        self.assertEqual( loc.is_dirty, True )

    def test_neighbor_locator(self):

        loc = self.loc

        # perform an update for the locator
        loc.update_status()

        self.assertEqual( loc.is_dirty, True )
        self.assertEqual( loc.manager.is_dirty, True)

        loc.update()

        self.assertEqual( loc.manager.is_dirty, False )
        self.assertEqual( loc.is_dirty, False )
        self.assertEqual( self.pa.is_dirty, False )

        x = self.x.astype(numpy.float32)
        y = self.y.astype(numpy.float32)
        z = self.z.astype(numpy.float32)
        
        cell_size = self.cell_size

        for i in range(self.np):

            nbrs = base.LongArray()

            brute_nbrs = ll.brute_force_neighbors(i,
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
