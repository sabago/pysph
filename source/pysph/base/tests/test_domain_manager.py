"""Tests for the various domain managers"""

import numpy

import unittest

# PySPH imports
import pysph.base.api as base
import pysph.solver.cl_utils as clu

import pysph.base.linked_list_functions as ll

if not clu.HAS_CL:
    try:
        import nose.plugins.skip as skip
        reason = "PyOpenCL not installed"
        raise skip.SkipTest(reason)
    except ImportError:
        pass

############################################################################
#`DomainManagerTestCase` class
############################################################################
class DomainManagerTestCase(unittest.TestCase):
    """General setup for all the domain managers"""

    def setUp(self):
        """The setup consists of ten particles with the following cellids:

        cellids = [7, 5, 6, 1, 8, 2, 5, 3, 8, 7]

        Nine cells are created by a manager that uses cells for
        binning. The constant cell size of the domain manager is used
        to create this special structure.
        
        """
        self.cl_precision = cl_precision = "single"

        self.x = x = numpy.array( [0.3, 0.6, 0.1, 0.3, 0.6,
                                   0.6, 0.6, 0.1, 0.6, 0.3] )

        self.y = y = numpy.array( [0.6, 0.3, 0.6, 0.1, 0.6,
                                   0.1, 0.3, 0.3, 0.6, 0.6] )

        self.np = len(x)

        # Construct the ParticleArray to be used with the Cython manager
        self.cy_pa = base.get_particle_array(
            name='test', cl_precision=cl_precision,
            x=x, y=y)

        # Construct the ParticleArray to be used with the OpenCL manager
        self.cl_pa = base.get_particle_array(
            name="test", cl_precision=self.cl_precision,
            x=x, y=y)

        # constant cell size to use
        self.const_cell_size = 0.25

        # constants based on the initial data
        self.mx, self.my = 0.1, 0.1
        self.Mx, self.My = 0.6, 0.6

        self.ncells = 9
        self.ncellsp1 = 10

        # perform the individual manager setup
        self._setup()

    def _setup(self):
        # create dummy managers
        self.cl_manager = base.LinkedListManager(
            arrays=[self.cl_pa,], cell_size=self.const_cell_size,
            with_cl=True)

        self.cy_manager = base.LinkedListManager(
            arrays=[self.cy_pa,], cell_size=self.const_cell_size,
            with_cl=False)
        

    def test_find_bounds(self):
        """DomainManagerTestCase: test_find_bounds

        This test is agnostic to the different types of managers.

        """ 
        # get the managers
        cy = self.cy_manager
        cl = self.cl_manager

        # test the with_cl flag
        self.assertEqual( cy.with_cl, False )
        self.assertEqual( cl.with_cl, True )

        # test the kernel scale factor
        self.assertEqual( cy.kernel_scale_factor, 2.0 )
        self.assertEqual( cl.kernel_scale_factor, 2.0 )

        # test the domain bounds
        self.assertAlmostEqual( cy.mx, self.mx, 6 )
        self.assertAlmostEqual( cl.mx, self.mx, 6 )

        self.assertAlmostEqual( cy.my, self.my, 6 )
        self.assertAlmostEqual( cl.my, self.my, 6 )

        self.assertAlmostEqual( cy.mz, 0.0, 6 )
        self.assertAlmostEqual( cl.mz, 0.0, 6 )

        # test the cell size
        self.assertAlmostEqual( cy.cell_size, self.const_cell_size, 6 )
        self.assertAlmostEqual( cl.cell_size, self.const_cell_size, 6 )        

############################################################################
#`LinkedListManagerTestCase` class
############################################################################
class LinkedListManagerTestCase(DomainManagerTestCase):

    def _setup(self):

        # create the linked list managers
        self.cl_manager = base.LinkedListManager(
            arrays=[self.cl_pa,], cell_size=self.const_cell_size,
            with_cl=True)

        self.cy_manager = base.LinkedListManager(
            arrays=[self.cy_pa,], cell_size=self.const_cell_size,
            with_cl=False)

    def test_update(self):
        """LinkedListManager:: test_update

        Test the Cython version of the LinkedListManager to coorectly
        produce the cellids, head and next arrays for the given
        particle distribution.

        """
        # get the manager
        cy = self.cy_manager

        cy.update()

        # test the updated data structures
        cellids = cy.cellids["test"]
        head = cy.head["test"]
        next = cy.Next["test"]

        _cellids = [7, 5, 6, 1, 8, 2, 5, 3, 8, 7]
        _head = [-1, 3, 5, 7, -1, 6, 2, 9, 8]
        _next = [-1, -1, -1, -1, -1, -1, 1, -1, 4, 0]

        for i in range(self.np):
            self.assertEqual( cellids[i], _cellids[i] )
            self.assertEqual( next[i], _next[i] )

        for i in range(self.ncells):
            self.assertEqual( head[i], _head[i] )        

    def test_iterator(self):
        """LinkedListManager: test_iterator

        The iterator should only return forward cells for each cell.

        """
        manager = self.cy_manager

        # update the data
        manager.update()

        ncx = manager.ncx
        ncy = manager.ncy
        ncz = manager.ncz

        for cell_nbrs in manager:

            # get the forward neighbors the brute force way
            cid = manager._current_cell - 1
            ix, iy, iz = ll.unflatten(cid, ncx, ncy)

            _cell_nbrs = []

            for i in range(ix -1, ix + 2):
                for j in range(iy -1, iy + 2):
                    for k in range(iz -1, iz + 2):

                        if ( (i >= 0) and (i < ncx) ):
                            if ( (j >= 0) and (j < ncy) ):
                                if ( (k >=0) and (k < ncz) ):
                                    _cid = i + j*ncx + k*ncx*ncy
                                    if _cid >= cid:
                                        _cell_nbrs.append(_cid)

            self.assertEqual(cell_nbrs, _cell_nbrs)

        # the current cell should be back to 0 after StopIteration
        self.assertEqual(manager._current_cell, 0)            

############################################################################
#`RadixSortManagerTestCase` class
############################################################################
class RadixSortManagerTestCase(DomainManagerTestCase):

    def _setup(self):

        self.cl_manager = base.RadixSortManager(
            arrays=[self.cl_pa,], cell_size=self.const_cell_size,
            with_cl=True)

        self.cy_manager = base.RadixSortManager(
            arrays=[self.cy_pa,], cell_size=self.const_cell_size,
            with_cl=False)

    def test_init_buffers(self):
        """RadixSortManager: test_init_buffers"""

        # get the domain managers
        cy = self.cy_manager
        cl = self.cl_manager

        # cellids and indices should be of length(np) = 9
        self.assertEqual( len(cy.cellids["test"]), self.np )
        self.assertEqual( len(cl.cellids["test"]), self.np )

        self.assertEqual( len(cy.indices["test"]), self.np )
        self.assertEqual( len(cl.indices["test"]), self.np )

        # number of cells should be 9 + 1
        self.assertEqual( len(cy.cell_counts["test"]), self.ncellsp1 )
        self.assertEqual( len(cl.cell_counts["test"]), self.ncellsp1 )

    def test_update(self):
        """RadixSortManager: test_update"""

        # get the Cython manager
        cy = self.cy_manager

        cy.update()

        # test the updated data structures
        cellids = cy.cellids["test"]
        indices = cy.indices["test"]
        cellc = cy.cell_counts["test"]

        # expected values
        sortedcellids = [1, 2, 3, 5, 5, 6, 7, 7, 8, 8]
        sortedindices = [3, 5, 7, 1, 6, 2, 0, 9, 4, 8]
        updatedcellc = [0, 0, 1, 2, 3, 3, 5, 6, 8, 10]

        # test the sortedcellids and sortedindices
        for i in range(self.np):
            self.assertEqual( cellids[i], sortedcellids[i] )
            self.assertEqual( indices[i], sortedindices[i] )

        # test the cell counts array
        for i in range(10):
            self.assertEqual( cellc[i], updatedcellc[i] )            
        
if __name__ == "__main__":
    unittest.main()
